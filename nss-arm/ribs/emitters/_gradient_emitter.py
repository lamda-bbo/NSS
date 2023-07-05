"""Provides the GradientEmitter."""
import itertools
import numpy as np
from numba import jit

from ribs.emitters._emitter_base import EmitterBase
from typing import List, Tuple, Callable


class GradientEmitter(EmitterBase):
    """Generates new solutions based on the gradient of the objective and measures.

    TODO: Write about the operator in more detail.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution, both when the archive is empty and afterwards. Note we
            assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        sigma_g (float): A step-size for the gradient in the gradient step. If measure
            gradients are used, sigma_g is the standard deviation of Gaussian noise
            used to sample gradient coefficients.
        line_sigma (float): The theta_2 parameter for a Iso+LineDD operator.
        measure_gradients (bool): Signals if measure gradients will be used.
        normalize_gradients (bool): Sets if gradients should be normalized before steps.
        operator_type (str): Either 'isotropic' or 'iso_line_dd' to mark the operator type 
            for intermediate operations. Defaults to 'isotropic'.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self,
                 archive,
                 x0,
                 selector='random',
                 sigma0=0.1,
                 sigma_g=0.05,
                 line_sigma=0.0,
                 measure_gradients=False,
                 normalize_gradients=False,
                 operator_type='isotropic',
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = archive.dtype(sigma0) if isinstance(
            sigma0, (float, np.floating)) else np.array(sigma0)
        self._sigma_g = archive.dtype(sigma_g)
        self._line_sigma = line_sigma
        self._use_isolinedd = operator_type != 'isotropic'
        self._measure_gradients = measure_gradients
        self._normalize_gradients = normalize_gradients

        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        if selector == 'random':
            self._selector = self.get_random_elites
        elif selector == 'mop1':
            self._selector = self.get_mop1_front_elites
        elif selector == 'mop2':
            self._selector = self.get_mop2_front_elites
        elif selector == 'mop3':
            self._selector = self.get_mop3_front_elites
        elif selector == 'nslc':
            self._selector = self.get_nslc_front_elites
        elif selector == 'curiosity':
            self._selector = self.get_curiosity_elites
        elif selector == 'edocs':
            self._selector = self.get_edocs_elites
        elif selector == 'nss':
            self._selector = self.get_nss_front_elites

        self._scores: np.ndarray = np.zeros(archive.get_storage_dims())

        self._last_parents: List[List[int]] = [[] for _ in range(batch_size)]

    def get_random_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_idxs, elites = self.archive.get_random_elites(n)
        return occupied_idxs, elites

    def get_curiosity_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        occupied_idxs = np.argsort(-self._scores[occupied_indices_cols])
        occupied_idxs = occupied_idxs[:n % occupied_len]
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate(
                [occupied_idxs] + [np.arange(0, occupied_len) for _ in range(n // occupied_len)]
            )
        self._rng.shuffle(occupied_idxs)
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    def get_nslc_front_elites(self, n: int, k: int = 60) -> Tuple[np.ndarray, List[Tuple]]:
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
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    def get_edocs_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_idxs: np.ndarray = np.array(self.archive.get_edocs_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    def get_mop1_front_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_idxs = np.array(self.archive.get_mop1_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    def get_mop2_front_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_idxs = np.array(self.archive.get_mop2_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    def get_mop3_front_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_idxs = np.array(self.archive.get_mop3_selected_id(n=n))
        if len(occupied_idxs) < n:
            occupied_idxs = np.concatenate((
                occupied_idxs,
                self.archive.get_random_elites(n - len(occupied_idxs))[0]
            ))
        self._rng.shuffle(occupied_idxs)
        occupied_idxs = occupied_idxs[:n]
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    def get_nss_front_elites(self, n: int) -> Tuple[np.ndarray, List[Tuple]]:
        occupied_indices_cols = self.archive.get_occupied_indices_cols()
        occupied_len = len(occupied_indices_cols[0])
        nbss_fronts: List[List[int]] = self.archive.get_nss_front()
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
        return occupied_idxs, self.archive.elites_with_occupied_idxs(occupied_idxs)

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
        return self._x0

    @property
    def sigma0(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution."""
        return self._sigma0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @staticmethod
    @jit(nopython=True)
    def _ask_solutions_numba(parents, line_gaussian, directions):
        """Numba helper for calculating solutions."""
        return parents + line_gaussian * directions

    @staticmethod
    @jit(nopython=True)
    def _ask_clip_helper(parents, noise, lower_bounds, upper_bounds):
        """Numba equivalent of np.clip."""
        return np.minimum(np.maximum(parents + noise, lower_bounds),
                          upper_bounds)

    def ask(self, grad_estimate=False):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        If the archive is empty, solutions are drawn from a (diagonal) Gaussian
        distribution centered at ``self.x0``. Otherwise, each solution is drawn
        from a distribution centered at a randomly chosen elite. In either case,
        the standard deviation is ``self.sigma0``.
    
        Args:
            grad_estimate(bool): A boolean signifying if this ask is for a gradient
                estimate or a sampling step. If this is a gradient estimate,
                a Jacobian should be passed back as gradient information.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        
        # On a gradient estimate, apply Gaussian noise to parents.
        if grad_estimate:
            if self.archive.empty:
                parents = np.expand_dims(self._x0, axis=0)
            else:
                self._last_parents, elites = self._selector(self._batch_size)
                self._last_parents = self._last_parents.reshape((self.batch_size, 1))
                parents = [
                    elite[0]
                    for elite in elites
                ]

            if self._use_isolinedd:
                noise = self._rng.normal(
                    scale=self._sigma0,
                    size=(self._batch_size, self.solution_dim),
                ).astype(self.archive.dtype)

                directions = [(self.archive.get_random_elite()[0] - parents[i])
                              for i in range(self._batch_size)]
                line_gaussian = self._rng.normal(
                    scale=self._line_sigma,
                    size=(self._batch_size, 1),
                ).astype(self.archive.dtype)

                solutions = self._ask_solutions_numba(np.asarray(parents), line_gaussian, 
                                                      np.asarray(directions))
                self._parents = self._ask_clip_helper(solutions, noise, self.lower_bounds,
                                     self.upper_bounds)
            else:
                noise = self._rng.normal(
                    scale=self._sigma0,
                    size=(self._batch_size, self.solution_dim),
                ).astype(self.archive.dtype)

                self._parents = self._ask_clip_helper(np.asarray(parents), noise,
                                             self.lower_bounds, self.upper_bounds)

            return self._parents
            
        if self._measure_gradients:
            noise = self._rng.normal(
                    scale=self._sigma_g,
                    size=self._jacobian.shape[:2],
                )
            noise[:, 0] = np.abs(noise[:, 0])
            noise = np.expand_dims(noise, axis=2)
            offsets = np.sum(np.multiply(self._jacobian, noise), axis=1)
            sols = offsets + self._parents

        else:
            # Transform the Jacobian 
            if len(self._jacobian.shape) == 3:
                self._jacobian = np.squeeze(self._jacobian[:,0:1,:], axis=1)
            sols = self._jacobian * self._sigma_g + self._parents

        return sols

    def tell(self, solutions, objective_values, behavior_values, jacobian=None, metadata=None):
        """Inserts entries into the archive.

        This base class implementation (in :class:`~ribs.emitters.EmitterBase`)
        simply inserts entries into the archive by calling
        :meth:`~ribs.archives.ArchiveBase.add`. It is enough for simple emitters
        like :class:`~ribs.emitters.GaussianEmitter`, but more complex emitters
        will almost certainly need to override it.

        Args:
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            jacobian (numpy.ndarray): Jacobian matrix for differentiable QD algorithms.           
            metadata (numpy.ndarray): 1D object array containing a metadata
                object for each solution.
        """
        metadata = itertools.repeat(None) if metadata is None else metadata
        if self._last_parents is None:
            parents_idxs = [([]) * len(self.archive.get_storage_dims())] * len(solutions)
        else:
            parents_idxs: List[Tuple[List[int]]] = [
                self.archive.indexs_with_occupied_idxs(pa) for pa in self._last_parents
            ]
        for i, (sol, obj, beh, meta) in enumerate(zip(solutions, objective_values,
                                                      behavior_values, metadata)):
            status, value = self.archive.add(sol, obj, beh, meta)
        
            index = self.archive.get_index(beh)

            if status:
                self._scores[parents_idxs[i]] += 1
            else:
                self._scores[parents_idxs[i]] -= 0.5

        if self._normalize_gradients and jacobian is not None:
            norms = np.linalg.norm(jacobian, axis=2)
            norms += 1e-8 # Make this configurable later
            norms = np.expand_dims(norms, axis=2)
            jacobian /= norms
             
        self._jacobian = jacobian
