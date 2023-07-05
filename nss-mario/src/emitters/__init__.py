"""pyribs-compliant emitters."""
import gin
import ribs

import numpy as np
from numba import jit
from ribs.archives import Elite
from ribs.archives._archive_base import readonly
import logging
import time
import itertools
from typing import Callable, List, Tuple

from src.emitters.map_elites_baseline_emitter import MapElitesBaselineEmitter

__all__ = [
    "GaussianEmitter",
    "ImprovementEmitter",
    "MapElitesBaselineEmitter",
]


logger = logging.getLogger(__name__)

SCORE_KEY = 'score'


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""

    def __init__(self,
                 archive,
                 selector: Callable[["GaussianEmitter", int], Tuple[np.ndarray, List[Elite]]],
                 x0,
                 sigma0,
                 bounds=None,
                 batch_size=64,
                 seed=None):

        super().__init__(
            archive,
            x0,
            sigma0,
            bounds,
            batch_size,
            seed,
        )

        self._selector = selector

        self._parents: List[List[int]] = [[] for _ in range(batch_size)]

        scores: np.ndarray = np.zeros(archive.get_storage_dims())
        self._set_archive_attr(SCORE_KEY, scores)

    def _set_archive_attr(self, key: str, value):
        setattr(self.archive, self.__class__.__name__ + '#' + key, value)

    def _get_archive_attr(self, key: str):
        return getattr(self.archive, self.__class__.__name__ + '#' + key)

    def _set_score(self, scores: np.ndarray):
        self._set_archive_attr(SCORE_KEY, scores)

    def _get_score(self) -> np.ndarray:
        return self._get_archive_attr(SCORE_KEY)

    @staticmethod
    @jit(nopython=True)
    def _ask_clip_helper(parents, noise, lower_bounds, upper_bounds):
        """Numba equivalent of np.clip."""
        return np.minimum(np.maximum(parents + noise, lower_bounds),
                          upper_bounds)

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
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_nslc_front_elites(self, n: int, k: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        nslc_fronts: List[np.ndarray] = self.archive.get_nslc_front(k, n)
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
    def get_edocs_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs: np.ndarray = np.array(self.archive.get_edocs_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    @gin.configurable
    def get_mop1_front_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
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
    def get_mop2_selected_id(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
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
    def get_mop3_selected_id(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
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
    def get_nss_front_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        occupied_idxs = np.array(self.archive.get_nss_selected_id(n % occupied_len), dtype=np.int)
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate(
                [occupied_idxs] + [np.arange(0, occupied_len) for _ in range(n // occupied_len)]
            )
        self._rng.shuffle(occupied_idxs)
        return readonly(occupied_idxs), self.archive.elites_with_occupied_idxs(occupied_idxs)

    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        If the archive is empty, solutions are drawn from a (diagonal) Gaussian
        distribution centered at ``self.x0``. Otherwise, each solution is drawn
        from a distribution centered at a randomly chosen elite. In either case,
        the standard deviation is ``self.sigma0``.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        if self.archive.empty:
            parents = np.expand_dims(self._x0, axis=0)
        else:
            start_time = time.time()
            self._parents, elites = self._selector(self, self._batch_size)
            self._parents = self._parents.reshape((self._batch_size, 1))
            self._last_selection_time = time.time() - start_time
            parents = [
                elite.sol
                for elite in elites
            ]

        noise = self._rng.normal(
            scale=self._sigma0,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        return self._ask_clip_helper(np.asarray(parents), noise,
                                     self.lower_bounds, self.upper_bounds)

    def tell(self, solutions, objective_values, behavior_values, metadata=None):
        scores = self._get_score()
        metadata = itertools.repeat(None) if metadata is None else metadata
        if self._parents is None:
            parents_idxs = [([]) * len(self.archive.get_storage_dims())] * len(solutions)
        else:
            parents_idxs: List[Tuple[List[int]]] = [
                self.archive.indexs_with_occupied_idxs(pa) for pa in self._parents
            ]
        for i, (sol, obj, beh, meta) in enumerate(zip(solutions, objective_values,
                                                      behavior_values, metadata)):
            status, value = self.archive.add(sol, obj, beh, meta)

            index = self.archive.get_index(beh)

            if status:
                scores[parents_idxs[i]] += 1
            else:
                scores[parents_idxs[i]] -= 0.5


@gin.configurable
class ImprovementEmitter(ribs.emitters.ImprovementEmitter):
    """gin-configurable version of pyribs ImprovementEmitter."""
