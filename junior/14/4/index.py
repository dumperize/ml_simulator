import torch


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    dist_ap = (anchor - positive).pow(2).sum(1).sqrt()
    dist_an = (anchor - negative).pow(2).sum(1).sqrt()

    concat = torch.vstack(
        (dist_ap - dist_an + margin, torch.zeros(dist_ap.size(dim=0))))
    max_concat, _ = torch.max(concat, 0)
    return torch.mean(max_concat)
