"""Provides the PGAEmitter."""
import logging

import gin
import numpy as np
from ribs.archives import Elite

from src.objectives.gym_control.td3 import TD3
from src.emitters import Emitter

from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


@gin.configurable
class PGAEmitter(Emitter):
    """Emitter based on the PG variation from PGA-ME."""

    def __init__(
        self,
        archive,
        selector: Callable[[Emitter, int], Tuple[np.ndarray, List[Elite]]],
        x0,
        sigma0,
        batch_size,
        bounds=None,
        seed=None,
    ):
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0

        self._selector = selector

        super().__init__(
            archive,
            len(self._x0),
            bounds,
            seed,
        )

        self._greedy_eval = None

        self._parents: List[List[int]] = [[] for _ in range(batch_size)]

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def greedy_eval(self):
        """Performance of the last evaluated greedy solution."""
        return self._greedy_eval

    def ask(self, td3: TD3):
        """Returns batch_size solutions.

        One of the solutions is the greedy solution. The other batch_size - 1
        solutions are created by randomly choosing elites from the archive and
        applying gradient ascent to them with TD3.

        When the archive is empty, we sample solutions from a Gaussian
        distribution centered at x0 with std sigma0.

        WARNING: Bounds are currently not enforced.
        """
        if self.archive.empty:
            logger.info("Sampling solutions from Gaussian distribution")
            return np.expand_dims(self._x0, axis=0) + self._rng.normal(
                scale=self._sigma0,
                size=(self._batch_size, self.solution_dim),
            ).astype(self.archive.dtype)
        else:
            logger.info("Sampling solutions with PG variation")
            occupied_idxs, elites = self._selector(self, self._batch_size - 1)
            occupied_idxs.resize((self._batch_size - 1, 1))
            self._parents = [[]] + occupied_idxs.tolist()
            pg_solutions = [
                td3.gradient_ascent(elite.sol)
                for elite in elites
            ]
            logger.info("Solutions with PG variation: %d", len(pg_solutions))
            return [td3.first_actor()] + pg_solutions

    def tell(self, solutions, objective_values, behavior_values, metadata=None):
        self._greedy_eval = objective_values[0]
        super().tell(solutions, objective_values, behavior_values, metadata, self._parents)
