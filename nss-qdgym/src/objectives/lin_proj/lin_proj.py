from typing import Callable, Optional
from dataclasses import dataclass

import gin
import numpy as np

from src.objectives.objective_base import ObjectiveBase
from src.objectives.objective_result import ObjectiveResult


@gin.configurable
@dataclass
class LinProjConfig:

    target_shift: float = 0.4

    initial_sol: float = None

    sol_dim: int = 100

    behavior_sol_dim: int = None

    bahavior_dim: int = 2

    def get_lower_bound(self):
        if self.behavior_sol_dim is None:
            return -self.sol_dim / self.bahavior_dim * 5.12
        else:
            return -self.behavior_sol_dim * 5.12


    def get_upper_bound(self):
        if self.behavior_sol_dim is None:
            return self.sol_dim / self.bahavior_dim * 5.12
        else:
            return self.behavior_sol_dim * 5.12


@gin.configurable
def get_lin_proj_lower_bound():
    return LinProjConfig().get_lower_bound()


@gin.configurable
def get_lin_proj_upper_bound():
    return LinProjConfig().get_upper_bound()


class LinProj(ObjectiveBase):

    config: LinProjConfig

    def __init__(self, config: LinProjConfig):
        super().__init__(config)

    def initial_solution(self, seed: Optional[int] = None) -> np.ndarray:
        if self.config.initial_sol is None:
            rng: np.random.Generator = np.random.default_rng(seed)
            return rng.normal(size=self.config.sol_dim)
        else:
            sol = np.empty((self.config.sol_dim,))
            sol.fill(self.config.initial_sol)
            return sol

    def distance(self, solution: np.ndarray, beta: float) -> float:

        worst_solution: np.ndarray = solution.reshape((-1, 2)).copy()
        worst_solution[:] = [-5.12, 5.12]
        worst_solution = worst_solution.reshape((self.config.bahavior_dim, -1))
        solution = solution.reshape((self.config.bahavior_dim, -1))

        target_shift = 5.12 * self.config.target_shift

        best_obj = 0.0
        average = np.expand_dims(np.average(worst_solution, axis=-1), axis=-1)
        worst_obj = np.sum(np.abs(worst_solution - average)) + beta * np.sum(np.abs(average - target_shift)) * self.config.sol_dim / self.config.bahavior_dim

        average = np.expand_dims(np.average(solution, axis=-1), axis=-1)
        raw_obj = np.sum(np.abs(solution - average)) + beta * np.sum(np.abs(average - target_shift)) * self.config.sol_dim / self.config.bahavior_dim

        return (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    def evaluate(self, solution: np.ndarray, n_evals: int, seed: Optional[int] = None) -> ObjectiveResult:

        assert self.config.sol_dim % self.config.bahavior_dim == 0
        assert self.config.behavior_sol_dim is None or self.config.behavior_sol_dim <= self.config.sol_dim // self.config.bahavior_dim

        returns = np.zeros(n_evals, dtype=float)
        returns[:] = self.distance(solution)

        clipped = solution.copy()
        clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
        clipped[clip_indices] = 5.12 / clipped[clip_indices]
        bcs = np.concatenate(
            [
                np.sum(clipped[
                    self.config.sol_dim * i // self.config.bahavior_dim :
                    self.config.sol_dim * (i + 1) // self.config.bahavior_dim
                ] if self.config.behavior_sol_dim is None else clipped[
                    self.config.sol_dim * i // self.config.bahavior_dim :
                    self.config.sol_dim * i // self.config.bahavior_dim + self.config.behavior_sol_dim
                ], keepdims=True)
                for i in range(self.config.bahavior_dim)
            ],
        )
        bcs = [bcs for _ in range(n_evals)]

        return ObjectiveResult.from_raw(returns=returns, bcs=np.asarray(bcs))
