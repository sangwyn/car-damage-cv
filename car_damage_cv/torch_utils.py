"""Training utilities retained from the original notebooks/scripts."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


def evaluate_mse(model: nn.Module, device: torch.device, dataloader: Iterable) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs).detach().cpu().numpy()
            values.append(np.mean((outputs - targets.numpy()) ** 2))
    mean_value = float(np.mean(values)) if values else 0.0
    print(f"Mean MSE: {mean_value}")
    return mean_value


def get_position_from_periods(iteration: int, cumulative_period: Sequence[int]) -> int:
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


class CosineAnnealingRestartLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        periods,
        restart_weights=(1,),
        eta_min=0,
        last_epoch=-1,
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        if len(self.periods) != len(self.restart_weights):
            raise ValueError("periods and restart_weights should have the same length.")
        self.cumulative_period = [sum(self.periods[0 : i + 1]) for i in range(len(self.periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min
            + current_weight
            * 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def get_scheduler(optimizer):
    return CosineAnnealingRestartLR(
        optimizer,
        periods=[10] * 20,
        restart_weights=[
            1,
            0.5,
            0.5,
            0.5,
            0.5,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        eta_min=1e-8,
    )


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, channels, 1, 1) * y + bias.view(1, channels, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, channels, _, _ = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, channels, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
