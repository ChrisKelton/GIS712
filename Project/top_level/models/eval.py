import os
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import jsonpickle

from Project.model_loaders.arcgis_utils import ArcGISImageClassifier
from Project.model_loaders.models import load_resnet50_hsg_aiml
from Project.model_loaders.load_esri_model import load_esri_model
from Project.top_level.utils.generate_datasets import Sen12MSDataset
from Project.top_level.utils.sen12ms_dataLoader import ClcDataPath, Seasons
from Project.top_level.utils.torch_model import accuracy_score
from Project.top_level.models.fsl_model import FslSemSeg


def generate_confusion_matrix_from_df(
    df: pd.DataFrame,
    out_path: Path,
    to_percentages: bool = True,
):
    plt.figure(figsize=(max((10, len(df) + 1)), max((7, len(df) - 3))))
    ax = sns.heatmap(df, annot=True, fmt=".2f")
    plt.xticks(np.arange(0.5, len(df.columns) + 0.5), list(df.columns), rotation=70, fontsize="large")
    plt.yticks(np.arange(0.5, len(df.columns) + 0.5), list(df.columns), fontsize="large")
    if to_percentages:
        for t in ax.texts:
            t.set_text(t.get_text() + "%")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def generate_confusion_matrix_from_array(
    confusion_mat: np.ndarray,
    class_names: list[str],
    img_out_path: Path,
    csv_out_path: Optional[Path] = None,
    normalize: bool = True,
    to_percentages: bool = True,
):
    if confusion_mat.shape[0] != confusion_mat.shape[1]:
        raise RuntimeError(f"confusion_mat must be a square matrix, got shape: '{confusion_mat.shape}'")
    if img_out_path.is_dir():
        raise RuntimeError(f"img_out_path must point to a file, not a directory: '{img_out_path}'")
    if csv_out_path is None:
        csv_out_path = img_out_path.parent / f"{img_out_path.stem}.csv"

    if normalize:
        for row in range(confusion_mat.shape[0]):
            sum_ = np.sum(confusion_mat[row, :])
            confusion_mat[row, :] /= sum_

    if to_percentages:
        confusion_mat *= 100

    df = pd.DataFrame(
        confusion_mat,
        index=class_names,
        columns=class_names,
    )
    df.to_csv(str(csv_out_path))
    generate_confusion_matrix_from_df(
        df=df,
        out_path=img_out_path,
        to_percentages=to_percentages,
    )


ClassNames: list[str] = [
    "ArtificialSurfaces",
    "AgriculturalAreas",
    "ForestNaturalAreas",
    "Wetlands",
    "WaterBodies",
    "NoData",
]


UniqueValsMap: dict[int, int] = {
    0: 5,  # no data
    1: 0,  # artificial surfaces
    2: 1,  # agricultural areas
    3: 2,  # forest and semi natural areas
    4: 3,  # wetlands
    5: 4,  # water bodies
    6: 5,  # no data
    255: 5,  # no data
}


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    label_criterion: nn.Module,
    features_similarity_criterion: nn.Module,
    n_classes: int,
    class_names: list[str],
    out_path: Path,
    out_path_prefix: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, float]:
    accumulated_loss_vals: list[float] = []
    embedded_features_loss_vals: list[float] = []
    supp_labels_loss_vals: list[float] = []
    query_label_loss_vals: list[float] = []
    accuracy_vals: list[float] = []
    query_accuracy_vals: list[float] = []
    confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    query_confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    for supp_imgs, supp_labels, query_img, query_label in tqdm(dataloader, desc="Testing over Dataset", position=0, leave=True):
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
            mask_pixel_wise_features=True,
            do_prediction=False,
        )
        query_embedded_masked_pixel_wise_features = model(
            img=query_img,
            mask_pixel_wise_features=True,
            do_prediction=False,
        )
        embedded_features_loss = features_similarity_criterion(
            supp_set_features=supp_embedded_masked_pixel_wise_features,
            query_set_features=query_embedded_masked_pixel_wise_features,
        )

        supp_pred = model(
            img=supp_imgs,
            mask_pixel_wise_features=False,
            do_prediction=True,
        )
        supp_labels_loss = label_criterion(supp_pred, supp_labels.long())

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

    img_name = "test-support-set-confusion-matrix.png"
    csv_name = "test-support-set-confusion-matrix.csv"
    if out_path_prefix is not None:
        img_name = f"{out_path_prefix}--{img_name}"
        csv_name = f"{out_path_prefix}--{csv_name}"
    generate_confusion_matrix_from_array(
        confusion_mat=confusion_mat,
        class_names=class_names,
        img_out_path=out_path / img_name,
        csv_out_path=out_path / csv_name,
    )

    if len(query_accuracy_vals) > 0:
        img_name= "test-query-confusion-matrix.png"
        csv_name = "test-query-confusion-matrix.png"
        if out_path_prefix is not None:
            img_name = f"{out_path_prefix}--{img_name}"
            csv_name = f"{out_path_prefix}--{csv_name}"
        generate_confusion_matrix_from_array(
            confusion_mat=query_confusion_mat,
            class_names=class_names,
            img_out_path=out_path / img_name,
            csv_out_path=out_path / csv_name,
        )

    results: dict[str, float] = {
        "accumulated_loss": np.mean(accumulated_loss_vals),
        "embedded_features_loss": np.mean(embedded_features_loss_vals),
        "supp_labels_loss": np.mean(supp_labels_loss_vals),
        "accuracy": np.mean(accuracy_vals),
        "query_label_loss": np.mean(query_label_loss_vals) if len(query_label_loss_vals) > 0 else 0,
        "query_accuracy": np.mean(query_accuracy_vals) if len(query_accuracy_vals) > 0 else 0,
    }

    return results


@torch.no_grad()
def evaluate_backbone(
    model: Union[nn.Module, ArcGISImageClassifier],
    dataloader: DataLoader,
    n_classes: int,
    out_path: Path,
    esri_model_kwargs: Optional[dict] = None,
    overwrite: bool = False,
):
    confusion_mat_img_out_path = out_path / "ConfusionMatrix.png"
    confusion_mat_csv_path = out_path / "confusion-mat.csv"
    if not overwrite and confusion_mat_img_out_path.exists() and confusion_mat_csv_path.exists():
        print(f"Confusion matrix outputs exist. Skipping...")
        return

    if esri_model_kwargs is None:
        esri_model_kwargs = {}

    confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    model.to(device)
    for img, label in tqdm(dataloader, position=0, leave=True):
        img = img.to(device)
        label = label.to(device)
        y_true = torch.ravel(label).to(torch.uint8).detach().tolist()
        out = torch.argmax(model(img,**esri_model_kwargs), dim=1)
        y_pred = torch.ravel(out).to(torch.uint8).detach().tolist()

        indices = np.column_stack([y_true, y_pred])
        unique_vals, cnts = np.unique(indices, return_counts=True, axis=0)
        for unique_val, cnt in zip(unique_vals, cnts):
            confusion_mat[UniqueValsMap[unique_val[0]], UniqueValsMap[unique_val[1]]] += cnt

    generate_confusion_matrix_from_array(
        confusion_mat=confusion_mat,
        class_names=ClassNames,
        img_out_path=confusion_mat_img_out_path,
        csv_out_path=confusion_mat_csv_path,
    )


def evaluate_fsl_model(
    model: FslSemSeg,
    dataloader: DataLoader,
    n_classes: int,
    class_names: list[str],
    out_path: Path,
    out_path_prefix: Optional[str] = None,
) -> dict[str, float]:
    accuracy_vals: list[float] = []
    confusion_mat: np.ndarray = np.zeros((n_classes, n_classes))
    for img, label in tqdm(dataloader, desc="Evaluating FSL Model", position=0, leave=True):
        img = img.to(device)
        label = label.to(device)

        pred = model(img=img, mask_pixel_wise_features=False, do_prediction=True)
        accuracy_vals.append(float(accuracy_score(label.detach(), torch.argmax(pred, dim=1).detach())))

        y_true = torch.ravel(label).to(torch.uint8).detach().tolist()
        y_pred = torch.ravel(torch.argmax(pred, dim=1)).to(torch.uint8).detach().tolist()

        indices = np.column_stack([y_true, y_pred])
        unique_vals, cnts = np.unique(indices, return_counts=True, axis=0)
        for unique_val, cnt in zip(unique_vals, cnts):
            confusion_mat[model.get_label_from_target_labels_map(unique_val[0]), unique_val[1]] += cnt

    img_name = "evaluate-confusion-matrix.png"
    csv_name = "evaluate-confusion-matrix.csv"
    if out_path_prefix is not None:
        img_name = f"{out_path_prefix}--{img_name}"
        csv_name = f"{out_path_prefix}--{csv_name}"
    generate_confusion_matrix_from_array(
        confusion_mat=confusion_mat,
        class_names=class_names,
        img_out_path=out_path / img_name,
        csv_out_path=out_path / csv_name,
    )

    return {"accuracy": np.mean(accuracy_vals)}


def main_test_backbone():
    ks: list[int] = [1, 3, 5]
    k = ks[0]

    support_set_csvs_base_path = ClcDataPath / f"support-set--{k}"
    if not support_set_csvs_base_path.exists():
        raise RuntimeError(f"'{support_set_csvs_base_path}' does not exist!")

    # esri_model = load_esri_model()
    # models = [esri_model]
    # model_names = ["esri-model"]
    # model_kwargs = [{"normalize": False}]

    models = [load_resnet50_hsg_aiml()]
    model_names = ["HSG-AIML"]
    model_kwargs = [{}]

    batch_size: int = 1
    seed: int = 42
    n_workers: int = 1
    pin_memory: bool = True if torch.cuda.is_available() and n_workers > 1 else False

    base_out_path = ClcDataPath / "model-tests--no-normalize"

    datasets_by_season_csv_paths: dict[str, dict[str, Path]] = {}
    seasons: list[str] = [Seasons.SPRING.value, Seasons.FALL.value]
    for season in seasons:
        print(f"Running '{season}'")
        support_set_csv_path = support_set_csvs_base_path / f"{season}--support-set.csv"
        train_csv_path = support_set_csvs_base_path / f"{season}--not-support-set--train.csv"
        val_csv_path = support_set_csvs_base_path / f"{season}--not-support-set--val.csv"

        datasets_by_season_csv_paths[season] = {
            "support-set": support_set_csv_path,
            "train": train_csv_path,
            "val": val_csv_path,
        }

        season_out_path = base_out_path / season
        for model, model_name, model_kwarg in zip(models, model_names, model_kwargs):
            print(f"\tModel: {model_name}")
            model_out_path = season_out_path / model_name
            for data_name, data_csv_path in datasets_by_season_csv_paths[season].items():
                print(f"\t\t{data_name}")
                data_out_path = model_out_path / data_name
                data_out_path.mkdir(exist_ok=True, parents=True)

                dataset = Sen12MSDataset(
                    data_base_path=ClcDataPath,
                    csv_path=data_csv_path,
                    batch_size=batch_size,
                    seed=seed,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=dataset.batch_size,
                    shuffle=dataset.shuffle,
                    num_workers=n_workers,
                    pin_memory=pin_memory,
                )
                evaluate_backbone(
                    model=model,
                    dataloader=dataloader,
                    n_classes=6,
                    out_path=data_out_path,
                    esri_model_kwargs=model_kwarg,
                )


def get_latest_model_path(base_path: Path, path_identifier: str) -> Path:
    paths = sorted(base_path.glob(path_identifier))
    epochs = [int(path.stem.split("--")[-1]) for path in paths]

    return paths[epochs.index(max(epochs))]


def main_test_fsl():
    n_target_classes = 6
    test_img = torch.zeros((1, 13, 256, 256)).to(device)

    use_mp: bool = True
    class_names: list[str] = ["ArtificialSurfaces", "AgriculturalAreas", "ForestNaturalAreas", "Wetlands", "WaterBodies", "NoData"]
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

    ks: list[int] = [5, 3, 1]

    batch_size: int = 8
    data_size: int = 1000
    seed: int = 3407
    if use_mp:
        n_workers: int = int(min(batch_size, len(os.sched_getaffinity(0)) // 4))
    else:
        n_workers: int = 0
    print(f"Running with '{n_workers}' workers.")
    pin_memory: bool = True if torch.cuda.is_available() and n_workers > 1 else False

    opposite_season_map: dict[str, str] = {
        Seasons.FALL.value: Seasons.SPRING.value,
        Seasons.SPRING.value: Seasons.FALL.value,
    }

    base_model_state_dicts_path = ClcDataPath / f"FSL-Training--batch_size-{batch_size}"
    if not base_model_state_dicts_path.exists():
        raise RuntimeError(f"{base_model_state_dicts_path} does not exist!")

    for k in ks:
        print(f"Evaluating {k}-Shot Learning")
        data_csvs_base_path = ClcDataPath / f"support-set--{k}"
        if not data_csvs_base_path.exists():
            raise RuntimeError(f"'{data_csvs_base_path}' does not exist!")

        few_shot_models_base_path = base_model_state_dicts_path / f"{k}-shot--{model_name}"
        for base_season, test_season in opposite_season_map.items():
            backbone = load_esri_model()
            model = FslSemSeg(
                backbone=backbone,
                n_target_classes=n_target_classes,
                target_labels=target_labels,
                test_img=test_img,
                target_labels_map=target_labels_map,
                train=False,
            )
            base_season_path = few_shot_models_base_path / base_season
            model_path = get_latest_model_path(base_season_path, "query*.pth")
            print(f"Using '{model_path}' to evaluate.")
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)

            out_path = base_season_path / f"test-on-season-{test_season}"
            out_path.mkdir(exist_ok=True, parents=True)

            test_csv_path = data_csvs_base_path / f"{test_season}--not-support-set--val.csv"

            test_dataset = Sen12MSDataset(
                data_base_path=ClcDataPath,
                csv_path=test_csv_path,
                batch_size=batch_size,
                data_size=data_size,
                seed=seed,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=test_dataset.batch_size,
                shuffle=test_dataset.shuffle,
                num_workers=n_workers,
                pin_memory=pin_memory,
            )
            results = evaluate_fsl_model(
                model=model,
                dataloader=test_dataloader,
                n_classes=n_target_classes,
                class_names=class_names,
                out_path=out_path,
            )
            print(f"Accuracy: {results['accuracy']*100:.2f}%")
            (out_path / "results.json").write_text(jsonpickle.dumps(results))


if __name__ == '__main__':
    # main_test_backbone()
    main_test_fsl()
