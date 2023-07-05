"""Provides IsoLineEmitter."""
import gin
import numpy as np
from numba import jit

from ribs.archives import Elite

from src.emitters import Emitter

from typing import Callable, List, Tuple, Union


@gin.configurable
class IsoLineEmitter(Emitter):
    """Emits solutions that are nudged towards other archive solutions.

    If the archive is empty, calls to :meth:`ask` will generate solutions from
    an isotropic Gaussian distribution with mean ``x0`` and standard deviation
    ``iso_sigma``. Otherwise, to generate each new solution, the emitter selects
    a pair of elites :math:`x_i` and :math:`x_j` and samples from

    .. math::

        x_i + \\sigma_{iso} \\mathcal{N}(0,\\mathcal{I}) +
            \\sigma_{line}(x_j - x_i)\\mathcal{N}(0,1)

    This emitter is based on the Iso+LineDD operator presented in `Vassiliades
    2018 <https://arxiv.org/abs/1804.03906>`_.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        iso_sigma (float): Scale factor for the isotropic distribution used when
            generating solutions.
        line_sigma (float): Scale factor for the line distribution used when
            generating solutions.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    """

    def __init__(self,
                 archive,
                 selector: Union[Callable[[Emitter, int], Tuple[np.ndarray, List[Elite]]], List[Callable[[Emitter, int], Tuple[np.ndarray, List[Elite]]]]],
                 x0,
                 iso_sigma=0.01,
                 line_sigma=0.2,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._iso_sigma = archive.dtype(iso_sigma)
        self._line_sigma = archive.dtype(line_sigma)

        self._selector = selector

        super().__init__(
            archive,
            len(self._x0),
            bounds,
            seed,
        )

        self._parents: List[List[int]] = [[] for _ in range(batch_size)]

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
        return self._x0

    @property
    def iso_sigma(self):
        """float: Scale factor for the isotropic distribution used when
        generating solutions."""
        return self._iso_sigma

    @property
    def line_sigma(self):
        """float: Scale factor for the line distribution used when generating
        solutions."""
        return self._line_sigma

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @staticmethod
    @jit(nopython=True)
    def _ask_solutions_numba(parents, iso_gaussian, line_gaussian, directions):
        """Numba helper for calculating solutions."""
        return parents + iso_gaussian + line_gaussian * directions

    @staticmethod
    @jit(nopython=True)
    def _ask_clip_helper(solutions, lower_bounds, upper_bounds):
        """Numba version of clip."""
        return np.minimum(np.maximum(solutions, lower_bounds), upper_bounds)

    def ask(self):
        """Generates ``batch_size`` solutions.

        If the archive is empty, solutions are drawn from an isotropic Gaussian
        distribution centered at ``self.x0`` with standard deviation
        ``self.iso_sigma``. Otherwise, each solution is drawn as described in
        this class's docstring.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        iso_gaussian = self._rng.normal(
            scale=self._iso_sigma,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        if self.archive.empty:
            solutions = np.expand_dims(self._x0, axis=0) + iso_gaussian
        else:
            if type(self._selector) == list or type(self._selector) == tuple:
                occupied_idxs0, elites0 = self._selector[0](self, self._batch_size)
                occupied_idxs1, elites1 = self._selector[1](self, self._batch_size)
                occupied_idxs: np.ndarray = np.concatenate([occupied_idxs0, occupied_idxs1])
                elites = elites0 + elites1
            else:
                occupied_idxs, elites = self._selector(self, self._batch_size * 2)
            occupied_idxs.resize((2, self._batch_size))
            self._parents = occupied_idxs.T
            parents = [
                elite.sol
                for elite in elites[:self._batch_size]
            ]
            directions = [(elite.sol - parents[i])
                          for i, elite in enumerate(elites[self._batch_size:])]
            line_gaussian = self._rng.normal(
                scale=self._line_sigma,
                size=(self._batch_size, 1),
            ).astype(self.archive.dtype)

            solutions = self._ask_solutions_numba(np.asarray(parents),
                                                  iso_gaussian, line_gaussian,
                                                  np.asarray(directions))

        return self._ask_clip_helper(solutions, self.lower_bounds,
                                     self.upper_bounds)

    def tell(self, solutions, objective_values, behavior_values, metadata=None):
        return super().tell(solutions, objective_values, behavior_values, metadata, self._parents)
