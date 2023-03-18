import torch


def contrastive_loss(
    x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, margin: float = 5.0
) -> torch.Tensor:
    """
    Computes the contrastive loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        x1 (torch.Tensor): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (torch.Tensor): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (torch.Tensor): Ground truth labels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The contrastive loss
    """
    euclidian_dist = (x1 - x2).pow(2).sum(1).sqrt()

    concat = torch.vstack(
        (margin - euclidian_dist, torch.zeros(euclidian_dist.size(dim=0))))

    pow_euclidian_dist = torch.pow(euclidian_dist, 2)
    max_concat, _ = torch.max(concat, 0)
    pow_max_contact = torch.pow(max_concat, 2)
    loss_arr = y * pow_euclidian_dist + (1-y) * pow_max_contact

    return torch.mean(loss_arr)


# x1 = torch.Tensor([[-0.9100, -5.9500, 4.2400, -3.0500], [-5.2000, -2.9800, -8.1000, -8.0700]])
# x2 = torch.Tensor([[-5.2100, -5.2900, -8.6100, -9.6100], [-2.0500, -7.8600, -9.9300, 2.0000]])
# y = torch.Tensor([0, 0])

# print(contrastive_loss(x1, x2, y, 15))
