"""Contains the GridArchive."""
import numpy as np
from numba import jit

from ribs.archives._archive_base import ArchiveBase

try:
    from pymoo.algorithms.moo.nsga2 import (NonDominatedSorting, calc_crowding_distance,
                                            randomized_argsort)
except ImportError:
    from pymoo.algorithms.nsga2 import (NonDominatedSorting, calc_crowding_distance,
                                        randomized_argsort)
from sklearn.cluster import KMeans
from typing import List, Tuple, Union, Optional


_EPSILON = 1e-6


class DominatorNumba:

    @staticmethod
    @jit(nopython=True)
    def non_surrounded_dominated_sort(dim, F, obj):

        def calc_surrounded_domination_matrics(dim, F):

            def get_relation(a, b):
                val = 0
                for i in range(len(a)):
                    if a[i] < b[i]:
                        if val == -1:
                            return 0
                        val = 1
                    elif b[i] < a[i]:
                        if val == 1:
                            return 0
                        val = -1
                return val

            dirs = np.ones((2 ** dim, dim))
            i = np.arange(dirs.shape[0])
            for j in range(dim):
                dirs[i >> j & 1 == 0, j] = -1
            max_dir = dirs.shape[0] - 1

            F = F.reshape((-1, dim))
            n = F.shape[0]

            Ms = np.zeros((dirs.shape[0], n, n))
            for k in range(dirs.shape[0] // 2):
                G = F * dirs[k]
                for i in range(n):
                    for j in range(i + 1, n):
                        rel = get_relation(G[i, :], G[j, :])
                        Ms[k, i, j] = rel
                        Ms[k, j, i] = -rel
                        Ms[max_dir - k, i, j] = -rel
                        Ms[max_dir - k, j, i] = rel

            return Ms

        Ms = calc_surrounded_domination_matrics(dim, F)

        n = Ms.shape[1]

        fronts = []

        if n == 0:
            return fronts

        n_ranked = 0

        is_dominating = [[[-1] for _ in range(2 ** dim)] for _ in range(n)]

        for tmp1 in is_dominating:
            for tmp2 in tmp1:
                tmp2.clear()

        n_dominated = np.zeros((n, 2 ** dim))

        current_front = []

        for i in range(n):
            for j in range(n):
                if obj[i] > obj[j]:
                    for k in range(2 ** dim):
                        if Ms[k, i, j] == 1:
                            is_dominating[i][k].append(j)
                            n_dominated[j, k] += 1

        for i in range(n):
            if (n_dominated[i] == 0).any():
                current_front.append(i)
                n_ranked += 1

        fronts.append(current_front)

        while n_ranked < n:

            next_front = []

            for i in current_front:
                for k in range(2 ** dim):
                    for j in is_dominating[i][k]:
                        if not (n_dominated[j] == 0).any():
                            n_dominated[j, k] -= 1
                            if (n_dominated[j] == 0).any():
                                next_front.append(j)
                                n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts

    @staticmethod
    @jit(nopython=True)
    def mop2_sort(F, obj):

        def calc_mop2_domination_matrics(F):

            def get_relation(a, b, oa, ob):
                if oa < ob:
                    return 1
                elif ob < oa:
                    return -1
                val = 0
                for i in range(len(a)):
                    if a[i] < b[i]:
                        if val == -1:
                            return 0
                        val = 1
                    elif b[i] < a[i]:
                        if val == 1:
                            return 0
                        val = -1
                return val

            n = F.shape[0]

            M = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    M[i, j] = get_relation(F[i, :], F[j, :], obj[i], obj[j])
                    M[j, i] = -M[i, j]

            return M

        M = calc_mop2_domination_matrics(F)

        n = M.shape[0]

        fronts = []

        if n == 0:
            return fronts

        n_ranked = 0

        is_dominating = [[-1] for _ in range(n)]

        for tmp in is_dominating:
            tmp.clear()

        n_dominated = np.zeros(n)

        current_front = []

        for i in range(n):

            for j in range(i + 1, n):
                rel = M[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                n_ranked += 1

        fronts.append(current_front)

        while n_ranked < n:

            next_front = []

            for i in current_front:
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts

class GridArchive(ArchiveBase):
    """An archive that divides each dimension into uniformly-sized bins.

    This archive is the container described in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_. It can be visualized as an
    n-dimensional grid in the behavior space that is divided into a certain
    number of bins in each dimension. Each bin contains an elite, i.e. a
    solution that `maximizes` the objective function for the behavior values in
    that bin.

    Args:
        dims (array-like of int): Number of bins in each dimension of the
            behavior space, e.g. ``[20, 30, 40]`` indicates there should be 3
            dimensions with 20, 30, and 40 bins. (The number of dimensions is
            implicitly defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objective values,
            and behavior values. We only support ``"f"`` / :class:`np.float32`
            and ``"d"`` / :class:`np.float64`.
    Raises:
        ValueError: ``dims`` and ``ranges`` are not the same length.
    """

    def __init__(self, dims, ranges, seed=None, dtype=np.float64):
        self._dims = np.array(dims)
        if len(self._dims) != len(ranges):
            raise ValueError(f"dims (length {len(self._dims)}) and ranges "
                             f"(length {len(ranges)}) must be the same length")

        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=len(self._dims),
            seed=seed,
            dtype=dtype,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self.dtype)
        self._interval_size = self._upper_bounds - self._lower_bounds

        self._boundaries = []
        for dim, lower_bound, upper_bound in zip(self._dims, self._lower_bounds,
                                                 self._upper_bounds):
            self._boundaries.append(
                np.linspace(lower_bound, upper_bound, dim + 1))

    def get_storage_dims(self) -> np.ndarray:
        return self._storage_dims

    def get_objective_values(self) -> np.ndarray:
        return self._objective_values

    def get_occupied_indices_cols(self) -> Tuple[List[int]]:
        return self._occupied_indices_cols

    def indexs_with_occupied_idxs(self, occupied_idxs: List[int]) -> Tuple[List[int]]:
        return tuple(np.array(self._occupied_indices_cols)[:, occupied_idxs].tolist())

    def elites_with_occupied_idxs(self, occupied_idxs: Union[List[int], np.ndarray]) -> List[Tuple]:
        idxs = [
            self._occupied_indices[occupied_idx]
            for occupied_idx in occupied_idxs
        ]
        return [
            (
                self._solutions[idx],
                self._objective_values[idx],
                self._behavior_values[idx],
                idx,
                self._metadata[idx],
            ) for idx in idxs
        ]

    def get_random_elites(self, n) -> Tuple[np.ndarray, List[Tuple]]:
        if self.empty:
            raise IndexError("No elements in archive.")

        random_idxs = []
        elites = []
        for _ in range(n):
            random_idx = self._rand_buf.get(len(self._occupied_indices))
            index = self._occupied_indices[random_idx]

            random_idxs.append(random_idx)
            elites.append((
                self._solutions[index],
                self._objective_values[index],
                self._behavior_values[index],
                index,
                self._metadata[index],
            ))

        return np.array(random_idxs), elites

    @staticmethod
    @jit(nopython=True)
    def _get_max_in_cluster_numba(objectives, kmeans_labels, n):
        first: np.ndarray = np.empty((n,), dtype=np.int64)
        first.fill(-1)
        second: np.ndarray = np.empty(n, dtype=np.int64)
        second.fill(-1)
        n_obj = objectives.shape[0]
        for i in range(n_obj):
            label = kmeans_labels[i]
            if first[label] == -1 or objectives[i] > objectives[first[label]]:
                second[label] = first[label]
                first[label] = i
            elif second[label] == -1 or objectives[i] > objectives[second[label]]:
                second[label] = i

        selection = second == -1
        second[selection] = first[selection]

        return first, second

    def get_nslc_front(self, k: int, n: Optional[int] = None) -> List[int]:
        objectives: np.ndarray = self._objective_values[self._occupied_indices_cols]
        behaviors: np.ndarray = self._behavior_values[self._occupied_indices_cols]

        is_higher = np.expand_dims(objectives, axis=-1) > objectives
        dist: np.ndarray = np.sqrt((
            (np.expand_dims(behaviors, axis=1) - behaviors) ** 2
        ).sum(axis=-1))
        knn: np.ndarray = np.argsort(dist)[:, :k]
        lqs: np.ndarray = np.take_along_axis(is_higher, knn, axis=-1).sum(axis=-1)
        ns: np.ndarray = np.take_along_axis(dist, knn, axis=-1).sum(axis=-1)
        return NonDominatedSorting().do(
            np.concatenate([np.expand_dims(-lqs, axis=-1), np.expand_dims(-ns, axis=-1)], axis=-1),
            n_stop_if_ranked=n
        )

    def get_edocs_selected_id(self, n) -> List[int]:
        if n > len(self._occupied_indices):
            return np.arange(len(self._occupied_indices))

        objectives: np.ndarray = self._objective_values[self._occupied_indices_cols]
        behaviors: np.ndarray = self._behavior_values[self._occupied_indices_cols]

        kmeans = KMeans(n_clusters=n, random_state=9).fit(behaviors)

        max_agents, max_agents_2 = self._get_max_in_cluster_numba(objectives, kmeans.labels_, n)
        selected_agents= []

        max_cluster = np.argmax(objectives[max_agents])

        for i in range(n):
            if i != max_cluster:
                # select the top1-agent
                if self._rand_buf.get(2):
                    selected_agents.append(max_agents[i])
                # select the top-2 agent
                else:
                    selected_agents.append(max_agents_2[i])
            else:
                selected_agents.append(max_agents[i])
        return selected_agents

    def get_mop1_selected_id(self, n: Optional[int] = None) -> List[int]:
        behaviors: np.ndarray = -self._behavior_values[self._occupied_indices_cols]

        survivors = []

        fronts = NonDominatedSorting().do(-behaviors, n_stop_if_ranked=n)

        for front in fronts:

            front = np.array(front)

            crowding_of_front = calc_crowding_distance(behaviors[front, :])

            if n is not None and len(survivors) + len(front) > n:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n - len(survivors))]
            else:
                I = np.arange(len(front))
            survivors.extend(front[I])

        return survivors

    def get_mop2_selected_id(self, eps: float = 0, layered: bool = False, n: Optional[int] = None) -> List[int]:
        objectives: np.ndarray = self._objective_values[self._occupied_indices_cols]
        behaviors: np.ndarray = -self._behavior_values[self._occupied_indices_cols]

        if layered:
            objectives //= eps
            eps = 0

        survivors = []

        fronts = DominatorNumba.mop2_sort(-behaviors, -objectives, eps)

        for front in fronts:

            front = np.array(front)

            crowding_of_front = calc_crowding_distance(behaviors[front, :])

            if n is not None and len(survivors) + len(front) > n:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n - len(survivors))]
            else:
                I = np.arange(len(front))
            survivors.extend(front[I])

        return survivors

    def get_mop3_selected_id(self, n: Optional[int] = None) -> List[int]:
        objectives: np.ndarray = self._objective_values[self._occupied_indices_cols]
        behaviors: np.ndarray = -self._behavior_values[self._occupied_indices_cols]

        survivors = []

        fronts = NonDominatedSorting().do(
            np.concatenate([np.expand_dims(-objectives, axis=-1), -behaviors], axis=-1),
            n_stop_if_ranked=n
        )

        for front in fronts:

            front = np.array(front)

            crowding_of_front = calc_crowding_distance(behaviors[front, :])

            if n is not None and len(survivors) + len(front) > n:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n - len(survivors))]
            else:
                I = np.arange(len(front))
            survivors.extend(front[I])

        return survivors

    def get_nss_front(self) -> List[List[int]]:
        objectives: np.ndarray = self._objective_values[self._occupied_indices_cols]
        behaviors: np.ndarray = self._behavior_values[self._occupied_indices_cols]

        idxs = np.argsort(-objectives)

        fronts = DominatorNumba.non_surrounded_dominated_sort(self._behavior_dim, behaviors[idxs], objectives[idxs])
        for i in range(len(fronts)):
            fronts[i] = idxs[fronts[i]].tolist()

        self._last_front_size = len(fronts[0])
        self._last_front_num = len(fronts)

        return fronts

    @property
    def dims(self):
        """(behavior_dim,) numpy.ndarray: Number of bins in each dimension."""
        return self._dims

    @property
    def lower_bounds(self):
        """(behavior_dim,) numpy.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(behavior_dim,) numpy.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self):
        """(behavior_dim,) numpy.ndarray: The size of each dim (upper_bounds -
        lower_bounds)."""
        return self._interval_size

    @property
    def boundaries(self):
        """list of numpy.ndarray: The boundaries of the bins in each dimension.

        Entry ``i`` in this list is an array that contains the boundaries of the
        bins in dimension ``i``. The array contains ``self.dims[i] + 1`` entries
        laid out like this::

            Archive bins:   | 0 | 1 |   ...   |    self.dims[i]    |
            boundaries[i]:  0   1   2   self.dims[i] - 1     self.dims[i]

        Thus, ``boundaries[i][j]`` and ``boundaries[i][j + 1]`` are the lower
        and upper bounds of bin ``j`` in dimension ``i``. To access the lower
        bounds of all the bins in dimension ``i``, use ``boundaries[i][:-1]``,
        and to access all the upper bounds, use ``boundaries[i][1:]``.
        """
        return self._boundaries

    @staticmethod
    @jit(nopython=True)
    def _get_index_numba(behavior_values, upper_bounds, lower_bounds,
                         interval_size, dims):
        """Numba helper for get_index().

        See get_index() for usage.
        """
        # Adding epsilon to behavior values accounts for floating point
        # precision errors from transforming behavior values. Subtracting
        # epsilon from upper bounds makes sure we do not have indices outside
        # the grid.
        behavior_values = np.minimum(
            np.maximum(behavior_values + _EPSILON, lower_bounds),
            upper_bounds - _EPSILON)

        index = (behavior_values - lower_bounds) / interval_size * dims
        return index.astype(np.int32)

    def get_index(self, behavior_values):
        """Returns indices of the entry within the archive's grid.

        First, values are clipped to the bounds of the behavior space. Then, the
        values are mapped to bins; e.g. bin 5 along dimension 0 and bin 3 along
        dimension 1.

        The indices can be used to access boundaries of a behavior value's bin.
        For example, the following retrieves the lower and upper bounds of the
        bin along dimension 0::

            idx = archive.get_index(...)  # Other methods also return indices.
            lower = archive.boundaries[0][idx[0]]
            upper = archive.boundaries[0][idx[0] + 1]

        See :attr:`boundaries` for more info.

        Args:
            behavior_values (numpy.ndarray): (:attr:`behavior_dim`,) array of
                coordinates in behavior space.
        Returns:
            tuple of int: The grid indices.
        """
        index = GridArchive._get_index_numba(behavior_values,
                                             self._upper_bounds,
                                             self._lower_bounds,
                                             self._interval_size, self._dims)
        return tuple(index)
