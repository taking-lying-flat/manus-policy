import math

import numpy as np
import torch
import torch.nn.functional as F


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    return (-(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std).sum(1, keepdim=True)


def categorical_log_density(actions, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    return log_probs.gather(-1, actions.long())


def get_flat_params_from(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def set_flat_params_to(model, flat_params):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_params[i:i + n].view(p.size()))
        i += n


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for p in net.parameters():
        if grad_grad:
            grads.append(p.grad.grad.view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)
