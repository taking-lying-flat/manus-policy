import numpy as np
import torch
import torch.nn as nn

from utils import get_flat_params_from, set_flat_params_to


def conjugate_gradients(
    fvp,
    b: torch.Tensor,
    nsteps: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        avp = fvp(p)
        alpha = rdotr / torch.dot(p, avp)
        x += alpha * p
        r -= alpha * avp
        new_rdotr = torch.dot(r, r)
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


# backtracking line search — shrink step until loss improves
def linesearch(
    model: nn.Module,
    loss_fn,
    x: torch.Tensor,
    fullstep: torch.Tensor,
    expected_improve_rate: torch.Tensor,
    max_backtracks: int = 10,
    accept_ratio: float = 0.1,
):
    fval = loss_fn(volatile=True)
    for stepfrac in 0.5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = loss_fn(volatile=True)
        actual = fval - newfval
        expected = expected_improve_rate * stepfrac
        ratio = actual / expected
        if ratio.item() > accept_ratio and actual.item() > 0:
            return True, xnew
    return False, x


def trpo_step(
    model: nn.Module,
    get_loss,
    get_kl,
    max_kl: float,
    damping: float,
) -> torch.Tensor:
    # gradient of surrogate loss
    loss = get_loss()
    loss_grad = torch.cat([g.view(-1) for g in torch.autograd.grad(loss, model.parameters())]).detach()

    # Fisher-vector product (Hessian of KL)
    def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
        kl = get_kl().mean()
        grad_kl = torch.cat([g.view(-1) for g in torch.autograd.grad(kl, model.parameters(), create_graph=True)])
        kl_v = (grad_kl * v).sum()
        grad_klv = torch.cat([g.contiguous().view(-1) for g in torch.autograd.grad(kl_v, model.parameters())])
        return grad_klv.detach() + v * damping

    # conjugate gradients → search direction
    stepdir = conjugate_gradients(fisher_vector_product, -loss_grad)
    # scale step to respect max_kl constraint
    shs = 0.5 * (stepdir * fisher_vector_product(stepdir)).sum(0, keepdim=True)
    lagrange = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lagrange[0]
    expected_improve = (-loss_grad * stepdir).sum(0, keepdim=True) / lagrange[0]

    # backtracking line search
    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(
        model, get_loss,
        prev_params, fullstep, expected_improve,
    )
    set_flat_params_to(model, new_params)
    return loss
