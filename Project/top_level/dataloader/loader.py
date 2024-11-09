import torch
from torch.utils.data import DataLoader
from Project.top_level.utils.sen12ms_dataLoader import SEN12MSDataset
from pathlib import Path


class GenericSen12MSDataloader(SEN12MSDataset, DataLoader):

    def __init__(self, base_dir: Path, ):
        super().__init__(str(base_dir))

