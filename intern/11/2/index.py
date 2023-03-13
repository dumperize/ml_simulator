from typing import List


def extend_matches(groups: List[tuple]) -> List[tuple]:
    l = []
    for group in groups:
        exist_s_idx = []
        for idx, s in enumerate(l):
            if any([x in s for x in group]):
                s.update(list(group))
                exist_s_idx.append(idx)
        if len(exist_s_idx) == 0:
            l.append(set(list(group)))
        if len(exist_s_idx) > 1:
            new_set = set()
            for idx in exist_s_idx:
                new_set.add(l[idx])
                del l[idx]
            l.append(new_set)

    return [tuple(sorted(el)) for el in sorted(l)]


if __name__ == "__main__":
    res = extend_matches([(1, 2, 3), (2, 3), (5, 3), (4, 6), (6, 7), (8, 9)])
    print(res)
