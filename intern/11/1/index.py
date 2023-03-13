from typing import List
from typing import Tuple
from itertools import combinations


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    l = []
    for x, y in pairs:
        exist_s_idx = []
        for idx, s in enumerate(l):
            if x in s or y in s:
                s.add(x)
                s.add(y)
                exist_s_idx.append(idx)
        if len(exist_s_idx) == 0:
            l.append(set([x, y]))
        if len(exist_s_idx) > 1:
            new_set = set()
            for idx in exist_s_idx:
                new_set.add(l[idx])
                del l[idx]
            l.append(new_set)

    new_list = []
    for s in l:
        new_list = new_list + list(combinations(sorted(list(s)), 2))

    return new_list


if __name__ == "__main__":
    res = extend_matches([(1, 2), (2, 3), (5, 3), (4, 6), (6, 7), (8, 9)])
    print(res)
