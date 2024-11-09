from pathlib import Path

import numpy as np
import pandas as pd

from Project.top_level.utils.sen12ms_dataLoader import ClcDataPath, Seasons
from torch.utils.data import Dataset
from typing import Optional, Callable
import torch
import rasterio


class Sen12MSDataset(Dataset):
    def __init__(
        self,
        data_base_path: Path,
        csv_path: Path,
        batch_size: int,
        bands: Optional[list[int]] = None,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_base_path = data_base_path
        self.csv_path = csv_path
        self.df = pd.read_csv(str(self.csv_path))
        self.df_idx = np.arange(0, len(self.df))
        self.rng = np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.df_idx)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bands = bands
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df_idx)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(index):
            index = index.tolist()

        series_ = self.df.iloc[index]
        image_path = self.data_base_path / str(series_["S2"])
        label_path = self.data_base_path / str(series_["LC"])

        with rasterio.open(image_path, 'r') as src:
            img = torch.Tensor(src.read(self.bands))
        with rasterio.open(label_path, 'r') as src:
            label = torch.Tensor(src.read(self.bands))

        sample = {"image": img, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["label"]


def split_dataset(csv_path: Path, train_split: float = 0.8, val_split: float = 0.2):
    df = pd.read_csv(str(csv_path), header=[0])
    entropies = list(df["entropy"])
    vals = np.unique(entropies)

    train_idxs: list[int] = []
    validation_idxs: list[int] = []
    flip_flop: bool = True
    flip_flop_ratio = max(int(np.round(train_split / val_split)), 1)
    flip_flop_cnt = 0
    for val in vals:
        idxs_into_df: np.ndarray = np.where(df["entropy"].where(lambda x: x == val) == val)[0]
        if len(idxs_into_df) > 1:
            n_train: int = int(len(idxs_into_df) * train_split)
            n_validation: int = len(idxs_into_df) - n_train

            if flip_flop:
                splitting_mod = int(np.round(n_train / n_validation))
            else:
                splitting_mod = n_train // n_validation
            flip_flop_cnt += 1
            if flip_flop_cnt == flip_flop_ratio:
                flip_flop = not flip_flop
                flip_flop_cnt = 0
            train_idxs_ = set(idxs_into_df[np.mod(np.arange(idxs_into_df.size), splitting_mod) != 0])
            validation_idxs_ = set(idxs_into_df).difference(train_idxs_)
        else:
            train_idxs_ = []
            validation_idxs_ = []
            if flip_flop:
                train_idxs_ = [idxs_into_df[0]]
            else:
                validation_idxs_ = [idxs_into_df[0]]
            flip_flop_cnt += 1
            if flip_flop_cnt == flip_flop_ratio:
                flip_flop = not flip_flop
                flip_flop_cnt = 0

        train_idxs.extend(list(train_idxs_))
        validation_idxs.extend(list(validation_idxs_))

    train_df = df.iloc[list(set(train_idxs))]
    train_df.sort_values("entropy", axis=0, ascending=False, inplace=True)
    validation_df = df.iloc[list(set(validation_idxs))]
    validation_df.sort_values("entropy", axis=0, ascending=False, inplace=True)

    train_csv_path = csv_path.parent / f"{csv_path.stem}--train.csv"
    train_df.to_csv(str(train_csv_path), index=None)
    validation_csv_path = csv_path.parent / f"{csv_path.stem}--val.csv"
    validation_df.to_csv(str(validation_csv_path), index=None)


def main():
    for k in [1, 3, 5]:
        support_set_base_csvs_path = ClcDataPath / f"support-set--{k}"
        for season in [Seasons.SPRING.value, Seasons.FALL.value]:
            not_support_set_csv_path = support_set_base_csvs_path / f"{season}--not-support-set.csv"
            split_dataset(not_support_set_csv_path)


if __name__ == '__main__':
    main()
