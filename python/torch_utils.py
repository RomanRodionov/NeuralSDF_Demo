import torch

def eikonal_loss(preds, points, grads=None):
    if grads is None:
        grads = torch.autograd.grad(
            outputs=preds,
            inputs=points,
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
        )[0]
    
    grad_norm = torch.norm(grads, dim=1)
    loss = torch.mean((grad_norm - 1.0) ** 2)

    return loss

def heat_loss(points, preds, grads=None, sample_pdfs=None, heat_lambda=8, in_mnfld=False):
    # Zimo Wang, & Li, T.-M. (2025). HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition. CVPR. 
    # https://zeamoxwang.github.io/HotSpot-CVPR25/

    if grads is None:
        grads = torch.autograd.grad(
            outputs=preds,
            inputs=points,
            grad_outputs=torch.ones_like(preds),
            create_graph=True,
            retain_graph=True,
        )[0]

    heat = torch.exp(-heat_lambda * preds.abs())

    if not in_mnfld:
        loss = 0.5 * heat**2 * (grads.norm(2, dim=-1) ** 2 + 1)
    else:
        loss = (0.5 * heat**2 * (grads.norm(2, dim=-1) ** 2 + 1)) - heat
    if sample_pdfs is not None:
        sample_pdfs = sample_pdfs.squeeze(-1)
        loss /= sample_pdfs
    loss = loss.mean()

    return loss