__all__ = ["plot_vals"]
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_vals(
    x_vals: list[Union[np.ndarray, list[Union[int, float]]]],
    y_vals: list[Union[np.ndarray, list[Union[int, float]]]],
    out_path: Path,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legends: Optional[list[str]] = None,
):
    assert len(x_vals) == len(y_vals), f"len(x_vals) != len(y_vals): {len(x_vals)} != {len(y_vals)}"
    if legends is not None:
        assert len(legends) == len(y_vals), f"len(legends) != length of values: {len(legends)} != {len(y_vals)}"
    fig = plt.figure(figsize=(7, 7))
    for idx, (x, y) in enumerate(zip(x_vals, y_vals)):
        plt.plot(x, y)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legends is not None:
        plt.legend(legends)
    plt.tight_layout()
    fig.savefig(str(out_path))
    plt.close()
