from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from utils import get_flat_params_from, set_flat_params_to

TrpoInfo = dict[str, float]


def conjugate_gradients(
    fvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    nsteps: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    if rdotr < residual_tol:
        return x

    for _ in range(nsteps):
        avp = fvp(p)
        denom = torch.dot(p, avp)
        if not torch.isfinite(denom) or torch.abs(denom) < 1e-12:
            break
        alpha = rdotr / denom
        x += alpha * p
        r -= alpha * avp
        new_rdotr = torch.dot(r, r)
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    fullstep: torch.Tensor,
    expected_improve_rate: torch.Tensor,
    max_backtracks: int = 10,
    accept_ratio: float = 0.1,
    kl_fn: Callable[[], torch.Tensor] | None = None,
    max_kl: float | None = None,
) -> tuple[torch.Tensor, float]:
    fval = loss_fn(eval_mode=True)
    for stepfrac in 0.5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(
            model=model,
            flat_params=xnew,
        )
        newfval = loss_fn(eval_mode=True)
        actual = fval - newfval
        expected = expected_improve_rate * stepfrac
        if expected.item() <= 0:
            continue
        ratio = actual / expected

        if kl_fn is not None and max_kl is not None:
            with torch.no_grad():
                kl = kl_fn().mean()
            if kl.item() > max_kl:
                continue

        if ratio.item() > accept_ratio and actual.item() > 0:
            return xnew, float(stepfrac)
    return x, 0.0


def trpo_step(
    model: nn.Module,
    get_loss: Callable[..., torch.Tensor],
    get_kl: Callable[[], torch.Tensor],
    max_kl: float,
    damping: float,
) -> tuple[torch.Tensor, TrpoInfo]:
    loss = get_loss()
    loss_before = loss.detach().item()
    loss_grad = torch.cat(
        [g.view(-1) for g in torch.autograd.grad(loss, model.parameters())]
    ).detach()

    def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
        kl = get_kl().mean()
        grad_kl = torch.cat(
            [
                g.view(-1)
                for g in torch.autograd.grad(
                    kl,
                    model.parameters(),
                    create_graph=True,
                )
            ]
        )
        kl_v = (grad_kl * v).sum()
        grad_klv = torch.cat(
            [
                g.reshape(-1)
                for g in torch.autograd.grad(
                    kl_v,
                    model.parameters(),
                )
            ]
        )
        return grad_klv.detach() + v * damping

    stepdir = conjugate_gradients(
        fvp=fisher_vector_product,
        b=-loss_grad,
    )
    shs = 0.5 * (stepdir * fisher_vector_product(stepdir)).sum()
    if not torch.isfinite(shs).all() or shs.item() <= 0:
        info: TrpoInfo = {
            "policy_improve": 0.0,
            "kl": 0.0,
            "line_search_step_frac": 0.0,
        }
        return loss, info

    lagrange = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lagrange
    expected_improve = (-loss_grad * stepdir).sum() / lagrange

    prev_params = get_flat_params_from(model)
    new_params, step_frac = linesearch(
        model=model,
        loss_fn=get_loss,
        x=prev_params,
        fullstep=fullstep,
        expected_improve_rate=expected_improve,
        kl_fn=get_kl,
        max_kl=max_kl,
    )
    set_flat_params_to(
        model=model,
        flat_params=new_params,
    )

    with torch.no_grad():
        loss_after = get_loss(eval_mode=True).item()
        final_kl = get_kl().mean().item()

    info = {
        "policy_improve": loss_before - loss_after,
        "kl": final_kl,
        "line_search_step_frac": step_frac,
    }
    return loss, info
