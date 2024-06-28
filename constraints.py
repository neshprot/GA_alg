from functools import partial

import numpy as np
from numpy.linalg import norm

from data import Protein


class Constraints:
    """
    Класс, соедржащий констрэинты
    """

    def __init__(self):
        self._constraints = []

    def add(self, f) -> None:
        """
        Добавляет новый коснтрэинт.
        Все функции должна обладать сигнатурой T -> bool.

        :param f: констрэинт.
        :return:
        """
        self._constraints.append(f)

    def check(self, x) -> bool:
        """
        Проверяет удовлетворение объекта всем констрэинтам

        :param x:
        :return: True, если все функции ограничения вернули True
        """
        for f in self._constraints:
            if not f(x):
                return False
        return True


def constraint_max_charge(p: Protein, max_charge: float) -> bool:
    return abs(p.charge) < max_charge


def constraint_distances(p: Protein, min_distance: float, coords: np.ndarray, positions_set) -> bool:
    assert len(p.genes) == len(coords)
    n = len(p.genes)
    for i in range(0, n):
        for j in range(i + 1, n):
            if (i + 1) in positions_set and (j + 1) in positions_set:
                if p.genes[i].charged and p.genes[j].charged:
                    dist = norm(coords[i] - coords[j])
                    if dist < min_distance:
                        return False
    return True


def constraint_max_num_changes(p: Protein, max_num_changes) -> bool:
    return p.num_changes <= max_num_changes


if __name__ == "__main__":
    from evolution import PositionsSetUnion
    from run_GA import coordinates
    import random
    from data import AMINOACIDS

    constraint = Constraints()


    f1 = partial(constraint_distances, min_distance=5.0, coords=coordinates, positions_set=PositionsSetUnion)
    f2 = partial(constraint_max_charge, max_charge=7)
    f3 = partial(constraint_max_num_changes, max_num_changes=6)

    constraint.add(f1)
    constraint.add(f2)
    constraint.add(f3)

    import random
    from data import AMINOACIDS

    sequence = random.choices(AMINOACIDS, k=200)
    protein = Protein(sequence)

    results_f = [f(protein) for f in [f1, f2, f3]]
    print(f"Constraint 1: {results_f[0]}")
    print(f"Constraint 2: {results_f[1]}")
    print(f"Constraint 3: {results_f[2]}")

    results_checker = constraint.check(protein)
    print(f"Constraints checker: {results_checker}")

    assert all(results_f) == results_checker

    print("OK!")
