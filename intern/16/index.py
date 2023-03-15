""" intern 16 """
from typing import List


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """ Function calc recall at k """
    all_t = [idx for idx, value in enumerate(labels) if value != 0]

    idx_scores_sorted = sorted(
        range(len(scores)),
        key=lambda x: scores[x],
        reverse=True
    )
    idx_scores_sorted_k = idx_scores_sorted[:k]

    intersect = set(idx_scores_sorted_k).intersection((all_t))

    return len(intersect) / len(all_t)


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """ Function calc precision at k """
    all_t = [idx for idx, value in enumerate(labels) if value != 0]

    idx_scores_sorted = sorted(
        range(len(scores)),
        key=lambda x: scores[x],
        reverse=True
    )
    idx_scores_sorted_k = idx_scores_sorted[:k]

    intersect = set(idx_scores_sorted_k).intersection((all_t))

    return len(intersect) / k


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """ Function calc specificity at k """
    all_f = [idx for idx, value in enumerate(labels) if value == 0]
    idx_scores_sorted = sorted(
        range(len(scores)),
        key=lambda x: scores[x],
        reverse=True
    )
    idx_scores_sorted_k = idx_scores_sorted[:k]
    intersect = set(all_f) - set(idx_scores_sorted_k)

    if len(all_f) == 0:
        return 0

    return len(intersect) / len(all_f)


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """ Function calc f1 score at k """
    recall = recall_at_k(labels, scores, k)
    precision = precision_at_k(labels, scores, k)

    if recall + precision == 0:
        return 0

    return 2*(recall * precision) / (recall + precision)


if __name__ == "__main__":
    exp_labels = [1, 0, 0, 1, 1]
    exp_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    print(precision_at_k(exp_labels, exp_scores, 1))
    print(recall_at_k(exp_labels, exp_scores, 1))
    print(specificity_at_k(exp_labels, exp_scores, 1))
    print(f1_at_k(exp_labels, exp_scores, 5))
