import collections
from codecs import namereplace_errors
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def get_parameter_count(model):
    return sum([p.numel() for p in model.parameters()])


def get_buffer_count(model):
    return sum([b.numel() for b in model.buffers()])


def get_weight_count(model):
    weight_count = 0
    for name, p in model.named_parameters():
        if "L" in name or "weight" in name:
            weight_count += p.numel()
    return weight_count


def get_bias_count(model):
    bias_count = 0
    for name, p in model.named_parameters():
        if "bias" in name:
            bias_count += p.numel()
    return bias_count


class LinearReparameterisedML(nn.Linear):
    """
    Re-parameterised Linear layer for compressed updates as per the "Mapping Learning"
    approach.
    """

    def __init__(
        self, linear, rank=None, rank_variance_prop=None, size_from_state_dict=None
    ):
        self.oc, self.ic = linear.weight.size()
        super(LinearReparameterisedML, self).__init__(self.ic, self.oc)
        delattr(self, "weight")
        if linear.bias is not None:
            self.bias.data = linear.bias.data
        else:
            self.bias = None

        r = None
        u, s, vh = torch.linalg.svd(linear.weight.data, full_matrices=False)

        if size_from_state_dict is not None:
            self.L = Parameter(torch.zeros(size_from_state_dict["L"].shape))
            self.register_buffer("R", torch.zeros(size_from_state_dict["R"].shape))

        else:
            if rank != "none" and rank_variance_prop == "none":
                r = rank
                assert r > 0, "Rank must be positive integers."
                if r > s.size(0):
                    r = s.size(0)
            elif rank_variance_prop != "none" and rank == "none":
                sum_s = torch.sum(s)
                for i in range(1, len(s)):
                    if torch.sum(s[:i]) >= rank_variance_prop * sum_s:
                        r = i
                        break
            else:
                raise ValueError(
                    "Exactly one of rank and rank_variance_prop must be specified, and the other 'none'"
                )

            self.L = Parameter(u[:, :r].mm(torch.diag(s[:r])))
            self.register_buffer("R", vh[:r, :])  # buffer registration freezes values

    def forward(self, x):
        self.weight = self.L.mm(self.R)
        return super(LinearReparameterisedML, self).forward(x)


def reparameterise_model_ML(model, rank=None, rank_variance_prop=None):
    model = deepcopy(model)
    for k, m in model.__dict__["_modules"].items():
        if isinstance(m, nn.Linear):
            model.__dict__["_modules"][k] = LinearReparameterisedML(
                m, rank, rank_variance_prop
            )
        elif isinstance(m, nn.ModuleList):
            for i, m2 in enumerate(m):
                model.__dict__["_modules"][k][i] = LinearReparameterisedML(
                    m2, rank, rank_variance_prop
                )
    return model


def reparameterise_model_ML_from_state_dict(model, state_dict):
    model = deepcopy(model)
    for k, m in model.__dict__["_modules"].items():
        if isinstance(m, nn.Linear):
            model.__dict__["_modules"][k] = LinearReparameterisedML(
                m,
                size_from_state_dict=dict(
                    L=state_dict[f"{k}.L"], R=state_dict[f"{k}.R"]
                ),
            )
        elif isinstance(m, nn.ModuleList):
            for i, m2 in enumerate(m):
                model.__dict__["_modules"][k][i] = LinearReparameterisedML(
                    m2,
                    size_from_state_dict=dict(
                        L=state_dict[f"{k}.{i}.L"], R=state_dict[f"{k}.{i}.R"]
                    ),
                )
    return model


def create_models_ML(model_coarse, model_fine, cfg):
    model_coarse = reparameterise_model_ML(
        model_coarse,
        cfg.federated.compress_rank,
        cfg.federated.compress_rank_variance_proportion,
    )
    if model_fine is not None:
        model_fine = reparameterise_model_ML(
            model_fine,
            cfg.federated.compress_rank,
            cfg.federated.compress_rank_variance_proportion,
        )
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    return (model_coarse, model_fine, trainable_parameters, optimizer)
