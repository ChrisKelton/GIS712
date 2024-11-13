from pathlib import Path
from Project.top_level.utils.visualization import plot_vals
from typing import Optional
import numpy as np


def extract_vals_from_logger(
    logger_txt_path: Path,
    out_path: Path,
    terms: Optional[list[str]] = None,
) -> tuple[dict[str, list[float]], dict[str, float], dict[str, list[float]], dict[str, float], dict[str, float]]:
    with open(str(logger_txt_path), 'r') as f:
        lines = [line.strip("\n") for line in f.readlines()]

    if terms is None:
        terms = [
            "accumulated_loss",
            "embedded_features_loss",
            "supp_labels_loss",
            "accuracy",
            "query_label_loss",
            "query_accuracy",
        ]
    n_terms_per_epoch = len(terms)
    val_first_results: dict[str, float] = {}
    try:
        val_first_result_idx = lines.index("Validate First Results:")
        results = lines[val_first_result_idx+1:val_first_result_idx+1+n_terms_per_epoch]
        results = [r.strip("\t").split(": ") for r in results]
        if len(terms) != len(results):
            raise RuntimeError(f"Mismatched number of terms in result with expected number of terms: '{len(results)}' != '{len(terms)}'")
        for result, term in zip(results, terms):
            val_first_results[term] = float(term[-1])
    except ValueError:
        pass

    def extract_results_from_epoch(epoch: int) -> tuple[dict[str, float], dict[str, float]]:
        result_idx = lines.index(f"Epoch {epoch}:")
        results = lines[result_idx+1:result_idx+1+n_terms_per_epoch]
        results = [r.strip("\t").replace(",", ":").split(": ") for r in results]
        if len(terms) != len(results):
            raise RuntimeError(f"Mismatched number of terms in result with expected number of terms: '{len(results)}' != '{len(terms)}'")

        train_out: dict[str, float] = {}
        val_out: dict[str, float] = {}
        for result, term in zip(results, terms):
            train_out[term] = float(result[1])
            val_out[term] = float(result[-1])

        return train_out, val_out

    train_results: dict[str, list[float]] = {}
    val_results: dict[str, list[float]] = {}
    epochs = [int(l.split("Epoch ")[-1].split(":")[0]) for l in lines if "Epoch" in l]
    for epoch in epochs:
        train_result, val_result = extract_results_from_epoch(epoch)
        for key, val in train_result.items():
            train_results.setdefault(key, []).append(val)
        for key, val in val_result.items():
            val_results.setdefault(key, []).append(val)

    x_train = np.asarray(epochs)
    for key, train_vals in train_results.items():
        val_vals = val_results[key]
        plot_path = out_path / f"{key}.png"
        plot_vals(
            x_vals=[x_train, x_train],  # only works b/c I was saving out validation stuff every epoch
            y_vals=[list(train_vals), list(val_vals)],
            out_path=plot_path,
            title=key.replace("_", " ").capitalize(),
            xlabel="Epoch",
            ylabel="Metric",
            legends=["train", "validation"],
        )


def main():
    logger_path: Path = Path(
        "/home/ckelton/repos/ncsu-masters/GIS712/Project/clc-data/FSL-Training--batch_size-16/5-shot--esri-model/ROIs1158_spring/logger.txt"
    )
    out_path = logger_path.parent / "plots"
    out_path.mkdir(exist_ok=True, parents=True)
    extract_vals_from_logger(logger_path, out_path)


if __name__ == '__main__':
    main()
