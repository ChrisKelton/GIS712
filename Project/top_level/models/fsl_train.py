import os
from pathlib import Path
from typing import Callable, Optional, Union, Any
import shutil

import jsonpickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random

from Project.model_loaders.load_esri_model import load_esri_model
from Project.top_level.models.eval import test_model, generate_confusion_matrix_from_array
from Project.top_level.models.fsl_model import FslSemSeg
from Project.top_level.utils.generate_datasets import FslSen12MSDataset, Sen12MSDataset
from Project.top_level.utils.sen12ms_dataLoader import ClcDataPath, Seasons
from Project.top_level.utils.torch_model import accuracy_score
from Project.top_level.utils.visualization import plot_vals

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def fsl_train_one_epoch(
    model: FslSemSeg,
    dataloader: DataLoader,
    label_criterion: nn.Module,
    features_similarity_criterion: nn.Module,
    optimizer: nn.Module,
    n_classes: int,
    class_names: list[str],
    base_out_path: Path,
    pseudo_label_generator: Optional[Callable] = None,
) -> tuple[nn.Module, dict[str, float]]:
    """
    Meta-Learning Few-Shot Learning Training Framework:

        support_imgs: support set
        support_labels: support set laels
        query_img: unlabeled query image

        1. generate pseudo-label for query_img
        2. mask out pixel-wise features from query_img using psuedo label for query_img
        3. encode masked pixel-wise features into some latent space using an encoder

        4. for support_img, support_label in zip(support_imgs, support_labels):
            4.1. compute pixel-wise features from model for the support_img and query_img
            4.2. mask out pixel-wise features from support_img using support_label
            4.3. Encode masked pixel-wise features into some latent space using an encoder
        5. for embedded_pixel_wise_feature_by_class in embedded_masked_query_pixel_wise_features:
            5.1. Compute cosine similarity between embedded_pixel_wise_feature_by_class and each
                  embedded_support_pixel_wise_features_by_class
            5.2. Since we don't really know what groupings between the masked embeddings are in the query image,
                  try to maximize similarity to the most similar class (maybe more than 1) and minimize similarity
                  to the other classes
            5.3. Accumulate loss
        6. Perform gradient backpropagation
        7. To avoid learning the wrong classes from our blind loss from step 5., predict a label for each support
                image and perform gradient backpropagation again to maintain stability of accuracy.


    Overall Training:
        Notes:
            - freeze backbone entirely
            - backbone is the same between support_img(s) & query_img
            - encoder is the same between support_img(s) & query_img
            - `x` indicates multiplication

        support_img(s) ---> backbone ---> pixel-wise features ---> x ---> masked pixel-wise features ---> encoder ---> embedded masked pixel-wise features ---/---> loss
                                                                   ^                                                                                       backpropagation
        support_label(s) ------------------------------------------|


        query_img --------> backbone ---> pixel-wise features ---> x ---> masked pixel-wise features ---> encoder ---> embedded masked pixel-wise features ---/---> loss
            |                                                      ^                                                                                       backpropagation
            ------> pseudo-label generation -----------------------|

        (freeze encoder head)
        support_img(s) ---> model ---/---> loss
                                    ^  backpropagation
        support_label(s) -----------|
        (unfreeze encoder head)

    Overall Testing:

        # test_img ---> backbone ---> pixel-wise features ---> encoder ---> embedded pixel-wise features ---> knn ---> prediction
        
        test_img ---> model ---> prediction
    """
    accumulated_loss_vals: list[float] = []
    embedded_features_loss_vals: list[float] = []
    supp_labels_loss_vals: list[float] = []
    query_label_loss_vals: list[float] = []
    accuracy_vals: list[float] = []
    query_accuracy_vals: list[float] = []
    confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    query_confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    for supp_imgs, supp_labels, query_img, query_label in tqdm(dataloader, desc="Iterating over Dataset", position=0, leave=True):
        optimizer.zero_grad()

        supp_imgs = supp_imgs.squeeze().to(device)
        supp_labels = supp_labels.squeeze().to(device)

        if len(supp_imgs.shape) == 5:
            supp_imgs = supp_imgs[0].squeeze(0)
        if len(supp_imgs.shape) == 3:
            supp_imgs = supp_imgs.unsqueeze(0)

        if len(supp_labels.shape) == 4:
            supp_labels = supp_labels[0].squeeze(0)
        if len(supp_labels.shape) == 2:
            supp_labels = supp_labels.unsqueeze(0)

        query_img = query_img.to(device)

        supp_embedded_masked_pixel_wise_features = model(
            img=supp_imgs,
            label=supp_labels,
            pseudo_label_generator=pseudo_label_generator,
            mask_pixel_wise_features=True,
            do_prediction=False,
        )
        query_embedded_masked_pixel_wise_features = model(
            img=query_img,
            pseudo_label_generator=pseudo_label_generator,
            mask_pixel_wise_features=True,
            do_prediction=False,
        )
        embedded_features_loss = features_similarity_criterion(
            supp_set_features=supp_embedded_masked_pixel_wise_features,
            query_set_features=query_embedded_masked_pixel_wise_features,
        )
        embedded_features_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with model.freeze_encoder_head():
            supp_pred = model(
                img=supp_imgs,
                mask_pixel_wise_features=False,
                do_prediction=True,
            )
            supp_labels_loss = label_criterion(supp_pred, supp_labels.long())
            supp_labels_loss.backward()
            optimizer.step()

        if query_label is not None:
            query_label = query_label.squeeze().to(device)
            if len(query_label.shape) == 2:
                query_label = query_label.unsqueeze(0)
            model.eval()
            query_pred = model(
                img=query_img,
                mask_pixel_wise_features=False,
                do_prediction=True,
            )
            query_label_loss = label_criterion(query_pred, query_label.long())
            query_accuracy = float(accuracy_score(query_label.detach(), torch.argmax(query_pred, dim=1).detach()))
            query_label_loss_vals.append(float(query_label_loss.detach()))
            query_accuracy_vals.append(query_accuracy)
            model.train()

            y_true = torch.ravel(query_label).to(torch.uint8).detach().tolist()
            y_pred = torch.ravel(torch.argmax(query_pred, dim=1)).to(torch.uint8).detach().tolist()

            indices = np.column_stack([y_true, y_pred])
            unique_vals, cnts = np.unique(indices, return_counts=True, axis=0)
            for unique_val, cnt in zip(unique_vals, cnts):
                query_confusion_mat[model.get_label_from_target_labels_map(unique_val[0]), unique_val[1]] += cnt

        embedded_features_loss_vals.append(float(embedded_features_loss.detach()))
        supp_labels_loss_vals.append(float(supp_labels_loss.detach()))
        accumulated_loss_vals.append(embedded_features_loss_vals[-1] + supp_labels_loss_vals[-1])
        
        accuracy_vals.append(float(accuracy_score(supp_labels.detach(), torch.argmax(supp_pred, dim=1).detach())))

        y_true = torch.ravel(supp_labels).to(torch.uint8).detach().tolist()
        y_pred = torch.ravel(torch.argmax(supp_pred, dim=1)).to(torch.uint8).detach().tolist()

        indices = np.column_stack([y_true, y_pred])
        unique_vals, cnts = np.unique(indices, return_counts=True, axis=0)
        for unique_val, cnt in zip(unique_vals, cnts):
            confusion_mat[model.get_label_from_target_labels_map(unique_val[0]), unique_val[1]] += cnt

    generate_confusion_matrix_from_array(
        confusion_mat=confusion_mat,
        class_names=class_names,
        img_out_path=base_out_path / "train-support-set-confusion-matrix.png",
        csv_out_path=base_out_path / "train-support-set-confusion-matrix.png"
    )

    if len(query_accuracy_vals) > 0:
        generate_confusion_matrix_from_array(
            confusion_mat=query_confusion_mat,
            class_names=class_names,
            img_out_path=base_out_path / "train-query-confusion-matrix.png",
            csv_out_path=base_out_path / "train-query-confusion-matrix.csv",
        )

    results: dict[str, float] = {
        "accumulated_loss": np.mean(accumulated_loss_vals),
        "embedded_features_loss": np.mean(embedded_features_loss_vals),
        "supp_labels_loss": np.mean(supp_labels_loss_vals),
        "accuracy": np.mean(accuracy_vals),
        "query_label_loss": np.mean(query_label_loss_vals) if len(query_label_loss_vals) > 0 else 0,
        "query_accuracy": np.mean(query_accuracy_vals) if len(query_accuracy_vals) > 0 else 0,
    }
    
    return model, results


def fsl_train(
    model: FslSemSeg,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_criterion: nn.Module,
    features_similarity_criterion: nn.Module,
    optimizer: nn.Module,
    base_out_path: Path,
    epochs: int = 10,
    epochs_to_test_val: int = 2,
    validate_first: bool = True,
    n_classes: int = 6,
    class_names: tuple[str, ...] = tuple(["ArtificialSurfaces", "AgriculturalAreas", "ForestNaturalAreas", "Wetlands", "WaterBodies", "NoData"]),
    *,
    overwrite_outputs: bool = False,
) -> tuple[
    nn.Module,
    dict[str, Union[int, float]],
    dict[str, Union[int, float]],
    dict[str, Union[int, float]],
    dict[str, Union[int, float]],
    dict[str, float],
]:
    if base_out_path.exists():
        files = [file for file in sorted(base_out_path.glob("*")) if file.is_file()]
        if len(files) > 0:
            if not overwrite_outputs:
                raise RuntimeError(f"Files exist at '{base_out_path}' and 'overwrite_outputs' not set.")
        else:
            shutil.rmtree(str(base_out_path))
    base_out_path.mkdir(exist_ok=True, parents=True)

    model.train()

    epoch_out_path = base_out_path / "epoch--0"
    epoch_out_path.mkdir(exist_ok=True, parents=True)
    validate_first_results: dict[str, float] = {}
    if validate_first:
        model.eval()
        results = test_model(
            model,
            val_loader,
            label_criterion=label_criterion,
            features_similarity_criterion=features_similarity_criterion,
            n_classes=n_classes,
            class_names=list(class_names),
            out_path=epoch_out_path,
            out_path_prefix="validate-first",
            overwrite=overwrite_outputs,
        )
        model.train()
        validate_first_results["accumulated_loss"] = results["accumulated_loss"]
        validate_first_results["embedded_features_loss"] = results["embedded_features_loss"]
        validate_first_results["supp_labels_loss"] = results["supp_labels_loss"]
        validate_first_results["accuracy"] = results["accuracy"]
        validate_first_results["query_label_loss"] = results["query_label_loss"]
        validate_first_results["query_accuracy"] = results["query_accuracy"]
        print(f"\nValidate First Results:\n"
              f"\tAccumulated Loss: {validate_first_results['accumulated_loss']:.3f}\n"
              f"\tEmbedded Features Loss: {validate_first_results['embedded_features_loss']:.3f}\n"
              f"\tSupport Labels Loss: {validate_first_results['supp_labels_loss']:.3f}\n"
              f"\tAccuracy: {validate_first_results['accuracy']:.3f}\n"
              f"\tQuery Label Loss: {validate_first_results['query_label_loss']:.3f}\n"
              f"\tQuery Accuracy: {validate_first_results['query_accuracy']:.3f}\n")

    train_results: dict[str, list[float]] = {
        "accumulated_loss": [],
        "embedded_features_loss": [],
        "supp_labels_loss": [],
        "accuracy": [],
        "query_label_loss": [],
        "query_accuracy": [],
    }
    best_train_result: dict[str, Union[int, float]] = {}
    best_query_train_result: dict[str, Union[int, float]] = {}

    x_val: list[int] = []
    val_results: dict[str, list[float]] = {
        "accumulated_loss": [],
        "embedded_features_loss": [],
        "supp_labels_loss": [],
        "accuracy": [],
        "query_label_loss": [],
        "query_accuracy": [],
    }
    best_val_result: dict[str, Union[int, float]] = {}
    best_query_val_result: dict[str, Union[int, float]] = {}
    for epoch in tqdm(range(epochs), desc="Training Model"):
        best_model: bool = False
        best_query_model: bool = False

        epoch_out_path = base_out_path / f"epoch--{epoch + 1}"
        epoch_out_path.mkdir(exist_ok=True, parents=True)
        
        model, results = fsl_train_one_epoch(
            model=model,
            dataloader=train_loader,
            label_criterion=label_criterion,
            features_similarity_criterion=features_similarity_criterion,
            optimizer=optimizer,
            n_classes=n_classes,
            class_names=list(class_names),
            base_out_path=epoch_out_path,
        )
        train_results["accumulated_loss"].append(results["accumulated_loss"])
        train_results["embedded_features_loss"].append(results["embedded_features_loss"])
        train_results["supp_labels_loss"].append(results["supp_labels_loss"])
        train_results["accuracy"].append(results["accuracy"])
        train_results["query_label_loss"].append(results["query_label_loss"])
        train_results["query_accuracy"].append(results["query_accuracy"])
        
        if train_results["accuracy"][-1] > best_train_result.get("accuracy", 0.0):
            best_model = True

        if train_results["query_label_loss"] != 0:
            if train_results["query_accuracy"][-1] > best_query_train_result.get("query_accuracy", 0.0):
                best_query_model = True
            
        if best_model:
            best_train_result["accumulated_loss"] = train_results["accumulated_loss"][-1]
            best_train_result["embedded_features_loss"] = train_results["embedded_features_loss"][-1]
            best_train_result["supp_labels_loss"] = train_results["supp_labels_loss"][-1]
            best_train_result["accuracy"] = train_results["accuracy"][-1]
            best_train_result["query_label_loss"] = train_results["query_label_loss"][-1]
            best_train_result["query_accuracy"] = train_results["query_accuracy"][-1]
            best_train_result["epoch"] = epoch + 1

        if best_query_model:
            best_query_train_result["accumulated_loss"] = train_results["accumulated_loss"][-1]
            best_query_train_result["embedded_features_loss"] = train_results["embedded_features_loss"][-1]
            best_query_train_result["supp_labels_loss"] = train_results["supp_labels_loss"][-1]
            best_query_train_result["accuracy"] = train_results["accuracy"][-1]
            best_query_train_result["query_label_loss"] = train_results["query_label_loss"][-1]
            best_query_train_result["query_accuracy"] = train_results["query_accuracy"][-1]
            best_query_train_result["epoch"] = epoch + 1

        if ((epoch + 1) % epochs_to_test_val == 0 or best_model) or epoch == 0:
            epoch_out_path.mkdir(exist_ok=True, parents=True)
            x_val.append(epoch + 1)
            model.eval()
            results = test_model(
                model,
                val_loader,
                label_criterion=label_criterion,
                features_similarity_criterion=features_similarity_criterion,
                n_classes=n_classes,
                class_names=list(class_names),
                out_path=epoch_out_path,
                overwrite=overwrite_outputs,
            )
            val_results["accumulated_loss"].append(results["accumulated_loss"])
            val_results["embedded_features_loss"].append(results["embedded_features_loss"])
            val_results["supp_labels_loss"].append(results["supp_labels_loss"])
            val_results["accuracy"].append(results["accuracy"])
            val_results["query_label_loss"].append(results["query_label_loss"])
            val_results["query_accuracy"].append(results["query_accuracy"])
            model.train()
            
            best_val_model: bool = False
            if val_results["accuracy"][-1] > best_val_result.get("accuracy", 0.0):
                best_val_model = True

            best_query_val_model: bool = False
            if val_results["query_label_loss"] != 0:
                if val_results["query_accuracy"][-1] > best_query_val_result.get("query_accuracy", 0.0):
                    best_query_val_model = True
            
            if best_val_model:
                best_val_result["accumulated_loss"] = val_results["accumulated_loss"][-1]
                best_val_result["embedded_features_loss"] = val_results["embedded_features_loss"][-1]
                best_val_result["supp_labels_loss"] = val_results["supp_labels_loss"][-1]
                best_val_result["accuracy"] = val_results["accuracy"][-1]
                best_val_result["query_label_loss"] = val_results["query_label_loss"][-1]
                best_val_result["query_accuracy"] = val_results["query_accuracy"][-1]
                best_val_result["epoch"] = epoch + 1
                
                model_out_path = base_out_path / f"{model_name}--{epoch + 1}.pth"
                torch.save(model.state_dict(), str(model_out_path))

            if best_query_val_model:
                best_query_val_result["accumulated_loss"] = val_results["accumulated_loss"][-1]
                best_query_val_result["embedded_features_loss"] = val_results["embedded_features_loss"][-1]
                best_query_val_result["supp_labels_loss"] = val_results["supp_labels_loss"][-1]
                best_query_val_result["accuracy"] = val_results["accuracy"][-1]
                best_query_val_result["query_label_loss"] = val_results["query_label_loss"][-1]
                best_query_val_result["query_accuracy"] = val_results["query_accuracy"][-1]
                best_query_val_result["epoch"] = epoch + 1

                query_model_out_path = base_out_path / f"query--{model_name}--{epoch + 1}.pth"
                torch.save(model.state_dict(), str(query_model_out_path))
               
            train_accumulated_loss = train_results["accumulated_loss"][-1]
            train_embedded_features_loss = train_results["embedded_features_loss"][-1]
            train_supp_labels_loss = train_results["supp_labels_loss"][-1]
            train_accuracy = train_results["accuracy"][-1]
            train_query_label_loss = train_results["query_label_loss"][-1]
            train_query_accuracy = train_results["query_accuracy"][-1]
            
            val_accumulated_loss = val_results["accumulated_loss"][-1]
            val_embedded_features_loss = val_results["embedded_features_loss"][-1]
            val_supp_labels_loss = val_results["supp_labels_loss"][-1]
            val_accuracy = val_results["accuracy"][-1]
            val_query_label_loss = val_results["query_label_loss"][-1]
            val_query_accuracy = val_results["query_accuracy"][-1]
            
            print(f"\n\nEpoch {epoch + 1}:\n"
                  f"\ttrain_accumulated_loss: {train_accumulated_loss:.3f}, val_accumulated_loss: {val_accumulated_loss:.3f}\n"
                  f"\ttrain_embedded_features_loss: {train_embedded_features_loss:.3f}, val_embedded_features_loss: {val_embedded_features_loss:.3f}\n"
                  f"\ttrain_supp_labels_loss: {train_supp_labels_loss:.3f}, val_supp_labels_loss: {val_supp_labels_loss:.3f}\n"
                  f"\ttrain_accuracy: {train_accuracy:.3f}, val_accuracy: {val_accuracy:.3f}\n"
                  f"\ttrain_query_label_loss: {train_query_label_loss:.3f}, val_query_label_loss: {val_query_label_loss:.3f}\n"
                  f"\ttrain_query_accuracy: {train_query_accuracy:.3f}, val_query_accuracy: {val_query_accuracy:.3f}\n")
                
    x_train = np.arange(1, len(train_results["accumulated_loss"]) + 1)
    
    loss_plot_path = base_out_path / f"{model_name}--accumulated-loss.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["accumulated_loss"], val_results["accumulated_loss"]],
        out_path=loss_plot_path,
        title="Accumulated Loss",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )
    
    loss_plot_path = base_out_path / f"{model_name}--embedded-features-loss.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["embedded_features_loss"], val_results["embedded_features_loss"]],
        out_path=loss_plot_path,
        title="Embedded Features Loss",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )
    
    loss_plot_path = base_out_path / f"{model_name}--support-labels-loss.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["supp_labels_loss"], val_results["supp_labels_loss"]],
        out_path=loss_plot_path,
        title="Support Labels Loss",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )
    
    acc_plot_path = base_out_path / f"{model_name}--accuracy.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["accuracy"], val_results["accuracy"]],
        out_path=acc_plot_path,
        title="Accuracy",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )

    loss_plot_path = base_out_path / f"{model_name}--query-label-loss.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["query_label_loss"], val_results["query_label_loss"]],
        out_path=loss_plot_path,
        title="Query Label Loss",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )

    acc_plot_path = base_out_path / f"{model_name}--query-accuracy.png"
    plot_vals(
        x_vals=[x_train, x_val],
        y_vals=[train_results["query_accuracy"], val_results["query_accuracy"]],
        out_path=acc_plot_path,
        title="Query Accuracy",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )
    
    optimizer_out_path = base_out_path / f"{model_name}--optimizer.pth"
    torch.save(optimizer.state_dict(), str(optimizer_out_path))
    
    print(f"Returning best model at '{model_out_path}'")
    model.load_state_dict(torch.load(str(model_out_path)))
    
    return (
        model,
        best_train_result,
        best_query_train_result,
        best_val_result,
        best_query_val_result,
        validate_first_results,
    )


class FeaturesSimilarityCriterion(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.embedding_loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, supp_set_features: torch.Tensor, query_set_features: torch.Tensor) -> torch.Tensor:
        prototypical_features = torch.mean(supp_set_features, dim=0)
        prototypical_features = prototypical_features.reshape(1, prototypical_features.shape[0], -1)
        query_set_features = query_set_features.reshape(query_set_features.shape[0], query_set_features.shape[1], -1)

        targets = np.ones((query_set_features.shape[1], prototypical_features.shape[1])) * -1
        targets[::6] = 1
        targets.reshape(query_set_features.shape[1], prototypical_features.shape[1])
        targets = torch.tensor(targets, device=supp_set_features.device)
        losses: torch.Tensor = torch.zeros_like(targets).to(targets.device)
        for query_idx in range(query_set_features.shape[1]):
            for supp_idx in range(prototypical_features.shape[1]):
                target = targets[query_idx, supp_idx].expand(1, *targets[query_idx, supp_idx].shape)
                losses[query_idx, supp_idx] = self.embedding_loss(
                    query_set_features[:, query_idx],
                    prototypical_features[:, supp_idx],
                    target,
                )

        return torch.sum(torch.mean(losses, dim=1))


class DiceLoss(nn.Module):
    def __init__(self, n_target_classes: int):
        super().__init__()
        self.n_target_classes = n_target_classes

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, do_softmax: bool = True) -> torch.Tensor:
        batch_size = prediction.shape[0]
        if target.shape[1] != self.n_target_classes:
            if target.shape[1] == 1:
                target = target.squeeze(1)
            if len(target.shape) != 3:
                raise RuntimeError(
                    f"target shape must be (B, H, W) to do one-hot encoding. Got target.shape: '{target.shape}'"
                )
            # do one-hot encoding
            target_shape = target.shape
            target = F.one_hot(
                target,
                num_classes=self.n_target_classes
            ).reshape(target.shape[0], self.n_target_classes, *target_shape[1:])
        if do_softmax:
            prediction = F.softmax(prediction)
        prediction = prediction.view(batch_size, -1)
        target = target.view(batch_size, -1)

        intersection = (prediction * target).sum(1)
        union = prediction.sum(1) + target.sum(1)
        dice = (2. * intersection) / (union + 1e-8)

        return dice.sum()


def set_random_seed(seed: int = 42, rank: int = 0, deterministic: bool = False):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

        # Additionally, some operations on a GPU are implemented stochastically for efficiency.
        # We want to ensure that all operations are deterministic on GPU (if used)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = True


def main():
    n_target_classes = 6
    test_img = torch.zeros((1, 13, 256, 256)).to(device)

    validate_first_results: bool = True
    overwrite_output: bool = True
    use_mp: bool = True

    target_labels = [0, 1, 2, 3, 4, 5]
    target_labels_map = {
        0: 5,  # no data (only occurs during the class-activation map pseudo label generator for query images)
        1: 0,  # artificial surfaces
        2: 1,  # agricultural areas
        3: 2,  # forest and semi natural areas
        4: 3,  # wetlands
        5: 4,  # water bodies
        6: 5,  # no data
    }
    model_name = "esri-model"

    label_criterion_name = "Dice"
    if label_criterion_name == "CrossEntropy":
        label_criterion = nn.CrossEntropyLoss()
    elif label_criterion_name == "Dice":
        label_criterion = DiceLoss(n_target_classes=n_target_classes)
    features_similarity_criterion = FeaturesSimilarityCriterion()

    learning_rate = 1e-3
    weight_decay = 3e-4

    # ks: list[int] = [1, 3, 5]
    ks: list[int] = [5, 3, 1]

    epochs: int = 100
    epochs_to_test_val: int = 1
    batch_size: int = 16
    data_size: int = 1000
    seed: int = 3407
    if use_mp:
        n_workers: int = int(min(batch_size, len(os.sched_getaffinity(0)) // 4))
    else:
        n_workers: int = 0
    print(f"Running with '{n_workers}' workers.")
    pin_memory: bool = True if torch.cuda.is_available() and n_workers > 1 else False

    set_random_seed(seed=seed, deterministic=False)

    main_config: dict[str, Any] = {
        "main": {
            "epochs": epochs,
            "epochs_to_test_val": epochs_to_test_val,
            "batch_size": batch_size,
            "data_size": data_size,
            "seed": seed,
            "n_workers": n_workers,
            "pin_memory": pin_memory,
        },
        "optimizer": {
            "name": "AdamW",
            "params": {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
        },
        "loss": {
            "labels": label_criterion_name,
            "features": "CosineSimilarity",
        },
        "n_target_classes": n_target_classes,
    }

    super_base_out_path = ClcDataPath / f"FSL-Training--batch_size-{batch_size}"
    super_base_out_path.mkdir(exist_ok=True, parents=True)

    (super_base_out_path / "config.json").write_text(jsonpickle.dumps(main_config, indent=2))

    for k in ks:
        print(f"Running {k}-Shot Learning")
        support_set_csvs_base_path = ClcDataPath / f"support-set--{k}"
        if not support_set_csvs_base_path.exists():
            raise RuntimeError(f"'{support_set_csvs_base_path}' does not exist!")

        base_out_path = super_base_out_path / f"{k}-shot--{model_name}"
        seasons: list[str] = [Seasons.SPRING.value, Seasons.FALL.value]
        for season in seasons:
            backbone = load_esri_model()
            model = FslSemSeg(
                backbone=backbone,
                n_target_classes=n_target_classes,
                target_labels=target_labels,
                test_img=test_img,
                target_labels_map=target_labels_map,
                train=True,
            )
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            season_out_base_path = base_out_path / season
            print(f"Running '{season}'")
            support_set_csv_path = support_set_csvs_base_path / f"{season}--support-set.csv"
            train_csv_path = support_set_csvs_base_path / f"{season}--not-support-set--train.csv"
            val_csv_path = support_set_csvs_base_path / f"{season}--not-support-set--val.csv"

            train_dataset = FslSen12MSDataset(
                data_base_path=ClcDataPath,
                support_set_csv_path=support_set_csv_path,
                query_set_csv_path=train_csv_path,
                query_set_size=data_size,
                batch_size=batch_size,
                seed=seed,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_dataset.batch_size,
                shuffle=train_dataset.shuffle,
                num_workers=n_workers,
                pin_memory=pin_memory,
            )

            val_dataset = FslSen12MSDataset(
                data_base_path=ClcDataPath,
                support_set_csv_path=support_set_csv_path,
                query_set_csv_path=val_csv_path,
                query_set_size=data_size,
                batch_size=batch_size,
                seed=seed,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=val_dataset.batch_size,
                shuffle=val_dataset.shuffle,
                num_workers=n_workers,
                pin_memory=pin_memory,
            )

            season_out_base_path.mkdir(exist_ok=True, parents=True)
            (
                best_model,
                best_train_result,
                best_query_train_result,
                best_val_result,
                best_query_val_result,
                validate_first_results,
            ) = fsl_train(
                model=model,
                model_name=model_name,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                label_criterion=label_criterion,
                features_similarity_criterion=features_similarity_criterion,
                optimizer=optimizer,
                base_out_path=season_out_base_path,
                epochs=epochs,
                epochs_to_test_val=epochs_to_test_val,
                validate_first=validate_first_results,
                n_classes=n_target_classes,
                overwrite_outputs=overwrite_output,
            )

            best_train_json_path = season_out_base_path / "best-train-result.json"
            best_train_json_path.write_text(jsonpickle.dumps(best_train_result, indent=2))

            best_query_train_json_path = season_out_base_path / "best-query-train-result.json"
            best_query_train_json_path.write_text(jsonpickle.dumps(best_query_train_result, indent=2))

            best_val_json_path = season_out_base_path / "best-val-result.json"
            best_val_json_path.write_text(jsonpickle.dumps(best_val_result, indent=2))

            best_query_val_json_path = season_out_base_path / "best-query-val-result.json"
            best_query_val_json_path.write_text(jsonpickle.dumps(best_query_val_result, indent=2))

            validate_first_results_json_path = season_out_base_path / "validate-first-results.json"
            validate_first_results_json_path.write_text(jsonpickle.dumps(validate_first_results, indent=2))


if __name__ == '__main__':
    main()
