from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm

def flatten(l):
    return [item for sublist in l for item in sublist]

def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return np.linalg.norm(documents - pointA, axis=1).reshape((documents.shape[0], 1))


def create_sw_graph(
    data: np.ndarray,
    num_candidates_for_choice_long: int = 10,
    num_edges_long: int = 2,
    num_candidates_for_choice_short: int = 10,
    num_edges_short: int = 2,
    use_sampling: bool = False,
    sampling_share: float = 0.05,
    dist_f: Callable = distance
) -> Dict[int, List[int]]:
    n = data.shape[0]
    dict_graph = np.zeros((n, n))

    for i in range(n):
        dist = dist_f(data[i], data[i:])
        dict_graph[i:, i] = dist.reshape(dict_graph[i:, i].shape)
        dict_graph[i, i:] = dist.reshape(dict_graph[i, i:].shape)

    res = {}

    for i in range(n):  
        vertex = dict_graph[i]
        vertex_idx = np.argsort(vertex)
        vertex_idx = vertex_idx[vertex_idx != i]

        idx_first = vertex_idx[:num_candidates_for_choice_short]
        idx_last = vertex_idx[-num_candidates_for_choice_long:]
    
        closest_n = np.random.choice(idx_first, num_edges_short, False)
        farest_n = np.random.choice(idx_last, num_edges_long, False)

        res[i] = list(set(np.hstack((closest_n, farest_n))))
    return res


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10,
        num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:


    dist_dict = {}
    def get_dist(candidats):
        dist = []
        for c in candidats:
            memo = dist_dict.get(c)
            if memo:
                dist.append(memo)
            else:
                res = dist_f(query_point, np.array([all_documents[c]])).flatten()[0]
                dist_dict[c] = res
                dist.append(res)
        return dist
    
    def all_equal(a1, a2):  
        return all([a in a1 for a in a2]) and all([a in a2 for a in a1])

    best_idxs = []

    candidats = []
    best_candidats = np.random.choice(list(graph_edges.keys()), num_start_points, False)

    while(not all_equal(candidats, best_candidats)):
        candidats = best_candidats
        neibor_candidats = flatten([graph_edges[idx] for idx in candidats])

        all_candidats = np.unique(np.hstack((candidats,neibor_candidats)))

        dists = get_dist(all_candidats)

        new_best_idxs = np.argsort(dists)[:search_k]
        best_candidats = [all_candidats[idx] for idx in new_best_idxs] 

    return best_candidats
