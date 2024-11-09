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

from Project.model_loaders.arcgis_utils import ArcGISImageClassifier
from Project.top_level.utils.generate_datasets import Sen12MSDataset
from Project.top_level.utils.sen12ms_dataLoader import ClcDataPath, Seasons
from Project.model_loaders.models import load_resnet50_hsg_aiml


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


def evaluate(
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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


def main():
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
                evaluate(
                    model=model,
                    dataloader=dataloader,
                    n_classes=6,
                    out_path=data_out_path,
                    esri_model_kwargs=model_kwarg,
                )


if __name__ == '__main__':
    main()
