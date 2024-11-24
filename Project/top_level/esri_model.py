import torch
from pathlib import Path
from Project.model_loaders.load_esri_model import load_esri_model


def main():
    esri_model = load_esri_model()


if __name__ == '__main__':
    main()
