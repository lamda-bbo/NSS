"""Provides the Emitter."""
from ribs.emitters import EmitterBase
from ribs.archives import Elite # , AddStatus
import numpy as np

from src.archives import GridArchive

import gin

import itertools
from typing import List, Tuple


SCORE_KEY = 'score'


def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr


@gin.configurable
class Emitter(EmitterBase):

    def __init__(self, archive: GridArchive, solution_dim, bounds, seed):
        super().__init__(archive, solution_dim, bounds)

        self.archive: GridArchive

        self._rng: np.random.Generator = np.random.default_rng(seed)

        self._last_selection_time: int = 0

        scores: np.ndarray = np.zeros(archive.get_storage_dims())
        self._set_archive_attr(SCORE_KEY, scores)

    def get_last_selection_time(self):
        return self._last_selection_time

    def _set_archive_attr(self, key: str, value):
        setattr(self.archive, self.__class__.__name__ + '#' + key, value)

    def _get_archive_attr(self, key: str):
        return getattr(self.archive, self.__class__.__name__ + '#' + key)

    def _set_score(self, scores: np.ndarray):
        self._set_archive_attr(SCORE_KEY, scores)

    def _get_score(self) -> np.ndarray:
        return self._get_archive_attr(SCORE_KEY)

    @gin.configurable
    def get_random_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs, elites = self.archive.get_random_elites(n)
        return readonly(occupied_idxs), elites

    @gin.configurable
    def get_mop1_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs = np.array(self.archive.get_mop1_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_mop2_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs = np.array(self.archive.get_mop2_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_mop3_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs = np.array(self.archive.get_mop3_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_nslc_elites(self, n: int, k: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        nslc_fronts: List[np.ndarray] = self.archive.get_nslc_fronts(k, n)
        occupied_idxs = []
        for front in nslc_fronts:
            if len(front) > n % occupied_len - len(occupied_idxs):
                self._rng.shuffle(front)
                occupied_idxs += front[:n % occupied_len - len(occupied_idxs)].tolist()
                break
            else:
                occupied_idxs += front.tolist()
        occupied_idxs = np.array(occupied_idxs, dtype=np.int)
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate(
                [occupied_idxs] + [np.arange(0, occupied_len) for _ in range(n // occupied_len)]
            )
        self._rng.shuffle(occupied_idxs)
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_curiosity_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        occupied_idxs = np.argsort(-self._get_score()[occupied_indices_cols])
        occupied_idxs = occupied_idxs[:n % occupied_len]
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate(
                [occupied_idxs] + [np.arange(0, occupied_len) for _ in range(n // occupied_len)]
            )
        self._rng.shuffle(occupied_idxs)
        if self._record_history:
            self._history['selected'].append(occupied_idxs)
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_edocs_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs = np.array(self.archive.get_edocs_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_nss_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        nbss_fronts: List[List[int]] = self.archive.get_nss_fronts()
        occupied_idxs = []
        for front in nbss_fronts:
            if len(front) > n % occupied_len - len(occupied_idxs):
                self._rng.shuffle(front)
                best_i = np.argmax(self.archive.get_objective_values()[occupied_indices_cols][front])
                front[0], front[best_i] = front[best_i], front[0]
                occupied_idxs += front[:n % occupied_len - len(occupied_idxs)]
                break
            else:
                occupied_idxs += front
        occupied_idxs = np.array(occupied_idxs, dtype=np.int)
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate(
                [occupied_idxs] + [np.arange(0, occupied_len) for _ in range(n // occupied_len)]
            )
        self._rng.shuffle(occupied_idxs)
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    def tell(self, solutions, objective_values, behavior_values, metadata=None, parents: List[List[int]] = None):
        metadata = itertools.repeat(None) if metadata is None else metadata
        if parents is None:
            parents_idxs = [([]) * len(self.archive.get_storage_dims())] * len(solutions)
        else:
            parents_idxs: List[Tuple[List[int]]] = [
                self.archive.indexs_with_occupied_idxs(pa) for pa in parents
            ]
        scores = self._get_score()
        for i, (sol, obj, beh, meta) in enumerate(zip(solutions, objective_values,
                                                      behavior_values, metadata)):
            status, value = self.archive.add(sol, obj, beh, meta)

            if status:
                scores[parents_idxs[i]] += 1
            else:
                scores[parents_idxs[i]] -= 0.5
