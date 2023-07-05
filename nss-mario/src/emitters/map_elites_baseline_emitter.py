import copy
from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase

from ribs.archives import Elite
from ribs.archives._archive_base import readonly
import logging
import itertools
from typing import List, Tuple, Callable


logger = logging.getLogger(__name__)

SCORE_KEY = 'score'


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    mazes.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
    """

    def __init__(self,
                 archive: ribs.archives.ArchiveBase,
                 selector: Callable[["MapElitesBaselineEmitter", int], Tuple[np.ndarray, List[Elite]]],
                 x0: np.ndarray,
                 bounds: Optional["array-like"] = None,
                 seed: int = None,
                 num_objects: int = gin.REQUIRED,
                 batch_size: int = gin.REQUIRED,
                 initial_population: int = gin.REQUIRED,
                 mutation_k: int = gin.REQUIRED):
        solution_dim = len(x0)
        super().__init__(archive, solution_dim, bounds)
        self._rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        assert solution_dim >= self.mutation_k

        self._selector = selector

        self.sols_emitted = 0

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

    @gin.configurable
    def get_random_elites(self, n: int) -> Tuple[np.ndarray, List[Elite]]:
        occupied_idxs, elites = self.archive.get_random_elites(n)
        return readonly(occupied_idxs), elites

    def ask(self):
        if self.sols_emitted < self.initial_population:
            self.sols_emitted += self.batch_size
            return self._rng.integers(self.num_objects,
                                     size=(self.batch_size, self.solution_dim))
        else:
            sols = []
            # select k spots randomly without replacement
            # and calculate the random replacement values
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self._rng.permuted(idx_array,
                                            axis=1)[:, :self.mutation_k]
            mutate_vals = self._rng.integers(self.num_objects,
                                            size=(self.batch_size,
                                                  self.mutation_k))

            self._parents, parent_sols = self._selector(self, self.batch_size)
            self._parents = self._parents.reshape((self.batch_size, 1))

            for i in range(self.batch_size):
                # parent_sol, *_ = self.archive.get_random_elite()
                parent_sol, *_ = parent_sols[i]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

            self.sols_emitted += self.batch_size
            return np.array(sols)

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
