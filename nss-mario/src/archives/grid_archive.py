"""Custom GridArchive."""
import gin
import numpy as np
import ribs.archives
from ribs.archives._archive_base import readonly

try:
    from pymoo.algorithms.moo.nsga2 import (NonDominatedSorting, calc_crowding_distance,
                                            randomized_argsort)
except ImportError:
    from pymoo.algorithms.nsga2 import (NonDominatedSorting, calc_crowding_distance,
                                        randomized_argsort)
from sklearn.cluster import KMeans
from numba import jit
from typing import List, Tuple, Union, Optional


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


@gin.configurable
class GridArchive(ribs.archives.GridArchive):
    """Based on pyribs GridArchive.

    This archive records history of its objectives and behavior values if
    record_history is True. Before each generation, call new_history_gen() to
    start recording history for that gen. new_history_gen() must be called
    before calling add() for the first time.
    """

    def __init__(self,
                 dims,
                 ranges,
                 seed=None,
                 dtype=np.float64,
                 record_history=True):
        super().__init__(dims, ranges, seed, dtype)
        self._record_history = record_history
        self._history = [] if self._record_history else None

        self._last_front_size = 0
        self._last_front_num = 0

    def get_storage_dims(self) -> np.ndarray:
        return self._storage_dims

    def get_objective_values(self) -> np.ndarray:
        return self._objective_values

    def get_occupied_indices_cols(self) -> Tuple[List[int]]:
        return self._occupied_indices_cols

    def indexs_with_occupied_idxs(self, occupied_idxs: List[int]) -> Tuple[List[int]]:
        return tuple(np.array(self._occupied_indices_cols)[:, occupied_idxs].tolist())

    def elites_with_occupied_idxs(self, occupied_idxs: Union[List[int], np.ndarray]) -> List[ribs.archives.Elite]:
        idxs = [
            self._occupied_indices[occupied_idx]
            for occupied_idx in occupied_idxs
        ]
        return [
            ribs.archives.Elite(
                readonly(self._solutions[idx]),
                self._objective_values[idx],
                readonly(self._behavior_values[idx]),
                idx,
                self._metadata[idx],
            ) for idx in idxs
        ]

    def best_elite(self):
        """Returns the best Elite in the archive."""
        if self.empty:
            raise IndexError("No elements in archive.")

        objectives = self._objective_values[self._occupied_indices_cols]
        idx = self._occupied_indices[np.argmax(objectives)]
        return ribs.archives.Elite(
            readonly(self._solutions[idx]),
            self._objective_values[idx],
            readonly(self._behavior_values[idx]),
            idx,
            self._metadata[idx],
        )

    def get_random_elites(self, n) -> Tuple[np.ndarray, List[ribs.archives.Elite]]:
        if self.empty:
            raise IndexError("No elements in archive.")

        random_idxs = []
        elites = []
        for _ in range(n):
            random_idx = self._rand_buf.get(len(self._occupied_indices))
            index = self._occupied_indices[random_idx]

            random_idxs.append(random_idx)
            elites.append(ribs.archives.Elite(
                readonly(self._solutions[index]),
                self._objective_values[index],
                readonly(self._behavior_values[index]),
                index,
                self._metadata[index],
            ))

        return np.array(random_idxs), elites

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

    def get_nss_selected_id(self, n: Optional[int] = None) -> List[List[int]]:
        objectives: np.ndarray = self._objective_values[self._occupied_indices_cols]
        behaviors: np.ndarray = self._behavior_values[self._occupied_indices_cols]

        idxs = np.argsort(-objectives)

        fronts = DominatorNumba.non_surrounded_dominated_sort(self._behavior_dim, behaviors[idxs], objectives[idxs])
        for i in range(len(fronts)):
            fronts[i] = idxs[fronts[i]].tolist()

        self._last_front_size = len(fronts[0])
        self._last_front_num = len(fronts)

        if n is None:
            return fronts

        survivors = []

        for front in fronts:

            front = np.array(front)

            crowding_of_front = calc_crowding_distance(np.concatenate((objectives[front].reshape(-1, 1), behaviors[front, :]), axis=-1))

            if n is not None and len(survivors) + len(front) > n:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n - len(survivors))]
            else:
                I = np.arange(len(front))
            survivors.extend(front[I])

        return survivors

    def get_last_front_size(self) -> int:
        return self._last_front_size

    def get_last_front_num(self) -> int:
        return self._last_front_num

    def new_history_gen(self):
        """Starts a new generation in the history."""
        if self._record_history:
            self._history.append([])

    def history(self):
        """Gets the current history."""
        return self._history

    def add(self, solution, objective_value, behavior_values, metadata=None):
        status, val = super().add(solution, objective_value, behavior_values,
                                  metadata)

        # Only save obj and BCs in the history.
        if self._record_history and status:
            self._history[-1].append([objective_value, behavior_values])

        return status, val
