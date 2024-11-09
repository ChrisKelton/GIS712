import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from Project.top_level.models.eval import evaluate
from tqdm import tqdm
from typing import Callable, Optional
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def fsl_train_one_epoch(
#    backbone: nn.Module,
#    encoder: nn.Module,
#    classifer_head: nn.Module,
    model: nn.Module,
    dataloader: DataLoader,
    label_criterion: nn.Module,
    features_similarity_criterion: nn.Module,
    optimizer: nn.Module,
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
            - backbone is the same between support_img(s) & query_img
            - encoder is the same between support_img(s) & query_img
            - `x` indicates multiplication

        support_img(s) ---> backbone ---> pixel-wise features ---> x ---> masked pixel-wise features ---> encoder ---> embedded masked pixel-wise features
                                                                   ^
        support_label(s) ------------------------------------------|


        query_img --------> backbone ---> pixel-wise features ---> x ---> masked pixel-wise features ---> encoder ---> embedded masked pixel-wise features
            |                                                      ^
            ------> pseudo-label generation -----------------------|

    Overall Testing:

        test_img ---> backbone ---> pixel-wise features ---> encoder ---> embedded pixel-wise features ---> knn ---> prediction
    """
    accumulated_loss_vals: list[float] = []
    embedded_features_loss_vals: list[float] = []
    supp_labels_loss_vals: list[float] = []
    accuracy_vals: list[float] = []
    for supp_imgs, supp_labels, query_img in tqdm(dataloader, desc="Iterating over Dataset", position=0, leave=True):
        optimizer.zero_grad()

        supp_imgs = supp_imgs.to(device)
        supp_labels = supp_labels.to(device)
        query_img = query_img.to(device)

        supp_embedded_masked_pixel_wise_features = model.encoder(model.backbone(supp_imgs, supp_labels))
        
        if pseudo_label_generator is not None:
            query_pseudo_label = pseudo_label_generator(query_img)
        else:
            query_pseudo_label = model.pseudo_label_generator(query_img)
        query_embedded_masked_pixel_wise_features = model.encoder(model.backbone(query_img, query_pseudo_label))

        embedded_features_loss = features_similarity_criterion(
            support_features=supp_embedded_masked_pixel_wise_features,
            query_features=query_embedded_masked_pixel_wise_features,
        )
        embedded_features_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # TODO: maybe the classifier head is a fully connected network or a CNN to transform to original image dimensions
        supp_pred = model.classifier_head(model.encoder(model.backbone(supp_imgs)))
        supp_labels_loss = label_criterion(
            target=supp_labels,
            pred=supp_pred,
        )
        supp_labels_loss.backward()
        optimizer.step()
        
        embedded_features_loss_vals.append(float(embedded_features_loss.detach()))
        supp_labels_loss_vals.append(float(supp_labels_loss.detach()))
        accumulated_loss_vals.append(embedded_features_loss_vals[-1] + supp_labels_loss_vals[-1])
        
        accuracy_vals.append(float(accuracy_score(supp_labels.detach(), supp_pred.detach())))

    results: dict[str, float] = {
        "accumulated_loss": np.mean(accumulated_loss_vals),
        "embedded_features_loss": np.mean(embedded_features_loss_vals),
        "supp_labels_loss": np.mean(supp_labels_loss_vals),
        "accuracy": np.mean(accuracy_vals),
    }
    
    return model, results
        


def fsl_train(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
    base_out_path: Path,
    epochs: int = 10,
    epochs_to_test_val: int = 2,
    validate_first: bool = True,
    n_classes: int = 6,
    *,
    overwrite_outputs: bool = False,
) -> tuple[nn.Module, dict[str, float], dict[str, float], dict[str, float]]:
    if base_out_path.exists():
        files = [file for file in sorted(base_out_path.glob("*")) if file.is_file()]
        if len(files) > 0 and not overwrite_outputs:
            raise RuntimeError(f"Files exist at '{base_out_path}' and 'overwrite_outputs' not set.")
    base_out_path.mkdir(exist_ok=True, parents=True)

    model.train()

    epoch_out_path = base_out_path / "epoch--0"
    epoch_out_path.mkdir(exist_ok=True, parents=True)
    validate_first_results: dict[str, float] = {}
    if validate_first:
        with torch.no_grad():
            results = test_model(
                model,
                val_loader,
                criterion=criterion,
                n_classes=n_classes,
                out_path=epoch_out_path,
                overwrite=overwrite_outputs,
            )
            validate_first_results["accumulated_loss"] = results["accumulated_loss"]
            validate_first_results["embedded_features_loss"] = results["embedded_features_loss"]
            validate_first_results["supp_labels_loss"] = results["supp_labels_loss"]
            validate_first_results["accuracy"] = results["accuracy"]
        print(f"Validate First Results:\n"
              f"\tAccumulated Loss: {validate_first_results['accumulated_loss']:.3f}\n"
              f"\tEmbedded Features Loss: {validate_first_results['embedded_features_loss']:.3f}\n"
              f"\tSupport Labels Loss: {validate_first_results['supp_labels_loss']:.3f}\n"
              f"\tAccuracy: {validate_first_results['accuracy']:.3f}")

    train_results: dict[str, list[float]] = {
        "accumulated_loss": [],
        "embedded_features_loss": [],
        "supp_labels_loss": [],
        "accuracy": [],
    }
    best_train_result: dict[str, float] = {}

    x_val: list[int] = []
    val_results: dict[str, list[float]] = {
        "accumulated_loss": [],
        "embedded_features_loss": [],
        "supp_labels_loss": [],
        "accuracy": [],
    }
    best_val_result: dict[str, float] = {}
    for epoch in tqdm(range(epochs), desc="Training Model"):
        best_model: bool = False

        epoch_out_path = base_out_path / f"epoch--{epoch + 1}"
        
        model, results = fsl_train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )
        train_results["accumulated_loss"].append(results["accumulated_loss"])
        train_results["embedded_features_loss"].append(results["embedded_features_loss"])
        train_results["supp_labels_loss"].append(results["supp_labels_loss"])
        train_results["accuracy"].append(results["accuracy"])
        
        if epoch == 0:
            best_model = True
        elif train_results["accuracy"][-1] > train_results["accuracy"][-2]:
            best_model = True
            
        if best_model:
            best_model = True
            best_train_result["accumulated_loss"] = train_results["accumulated_loss"][-1]
            best_train_result["embedded_features_loss"] = train_results["embedded_features_loss"][-1]
            best_train_result["supp_labels_loss"] = train_results["supp_labels_loss"][-1]
            best_train_result["accuracy"] = train_results["accuracy"][-1]
            
        if ((epoch + 1) % epochs_to_test_val == 0 or best_model) or epoch == 0:
            epoch_out_path.mkdir(exist_ok=True, parents=True)
            x_val.append(epoch + 1)
            with torch.no_grad():
                results = test_model(
                    model,
                    val_loader,
                    criterion=criterion,
                    n_classes=n_classes,
                    out_path=epoch_out_path,
                    overwrite=overwrite_outputs,
                )
                val_results["accumulated_loss"] = results["accumulated_loss"]
                val_results["embedded_features_loss"] = results["embedded_features_loss"]
                val_results["supp_labels_loss"] = results["supp_labels_loss"]
                val_results["accuracy"] = results["accuracy"]
            
            best_val_model: bool = False
            if epoch == 0:
                best_val_model = True
            elif val_results["accuracy"][-1] > val_results["accuracy"][-2]:
                best_val_model = True
            
            if best_val_model:
                best_val_result["accumulated_loss"] = val_results["accumulated_loss"][-1]
                best_val_result["embedded_features_loss"] = val_results["embedded_features_loss"][-1]
                best_val_result["supp_labels_loss"] = val_results["supp_labels_loss"][-1]
                best_val_result["accuracy"] = val_results["accuracy"]
                
                model_out_path = base_out_path / f"{model_name}--{epoch + 1}.pth"
                torch.save(model.state_dict(), str(model_out_path))
               
            train_accumulated_loss = train_results["accumulated_loss"][-1]
            train_embedded_features_loss = train_results["embedded_features_loss"][-1]
            train_supp_labels_loss = train_results["supp_labels_loss"][-1]
            train_accuracy = train_results["accuracy"][-1]
            
            val_accumulated_loss = val_results["accumulated_loss"][-1]
            val_embedded_features_loss = val_results["embedded_features_loss"][-1]
            val_supp_labels_loss = val_results["supp_labels_loss"][-1]
            val_accuracy = val_results["accuracy"][-1]
            
            print(f"\n\nEpoch {epoch}:\n"
                  f"\ttrain_accumulated_loss: {train_accumulated_loss:.3f}, val_accumulated_loss: {val_accumulated_loss:.3f}\n"
                  f"\ttrain_embedded_features_loss: {train_embedded_features_loss:.3f}, val_embedded_features_loss: {val_embedded_features_loss:.3f}\n"
                  f"\ttrain_supp_labels_loss: {train_supp_labels_loss:.3f}, val_supp_labels_loss: {val_supp_labels_loss:.3f}\n"
                  f"\ttrain_accuracy: {train_accuracy:.3f}, val_accuracy: {val_accuracy:.3f}")
                
    x_train = np.arange(1, len(train_results["loss"]) + 1)
    
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
        out_path=loss_plot_path,
        title="Accuracy",
        xlabel="Epoch",
        ylabel="Metric",
        legends=["train", "validation"],
    )
    
    optimizer_out_path = base_out_path / f"{model_name}--optimizer.pth"
    torch.save(optimizer.state_dict(), str(optimizer_out_path))
    
    print(f"Returning best model at '{model_out_path}'")
    model.load_state_dict(torch.load(str(model_out_path)))
    
    return model, best_train_results, best_val_result, validate_first_results


def main():
    model = some_model()


if __name__ == '__main__':
    main()
