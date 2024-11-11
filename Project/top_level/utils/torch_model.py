__all__ = ["print_number_of_params", "get_n_params", "freeze_layers", "accuracy_score"]
import numpy as np
import torch
import torch.nn as nn


def get_n_params(model: nn.Module) -> int:
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_params])


def print_number_of_params(model: nn.Module):
    n_params = get_n_params(model)
    print(f"Number of Parameters: {n_params / 1000}K")


def freeze_layers(model: nn.Module, n_layers_to_not_freeze: int = 0) -> nn.Module:
    params_to_freeze = len(list(model.parameters())) - n_layers_to_not_freeze
    for idx, param in enumerate(model.parameters()):
        if idx == params_to_freeze:
            break
        param.requires_grad = False

    return model


def accuracy_score(target: torch.Tensor, pred: torch.Tensor) -> float:
    diff = target - pred
    correct = torch.sum(diff == 0)
    incorrect = torch.sum(diff != 0)

    return float(correct / (correct + incorrect))
