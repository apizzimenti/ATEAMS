
import numpy as np
import warnings
import sys
import time

from math import comb

from ..arithmetic import Twist, ReducedKernelSample, ComputePersistencePairs
from ..common import Matrices, TooSmallWarning, NumericalInstabilityWarning, FINT


class InvadedCluster():
	_name = "InvadedCluster"
	
	def __init__(
			self, C, dimension=1, field=2, initial=None, full=False, stop=lambda: 1,
			tries=64, **kwargs
		):
		"""
		Initializes the plaquette invaded-cluster algorithm on the provided
		integer complex, detecting percolation in the `homology`-th homology
		group.

		Args:
			C (Complex): The `Complex` object on which we'll be running experiments.
			dimension (int=1): The dimension of cells on which we're percolating.
			field (int=2): Field characteristic.
			initial (np.array): A vector of spin assignments to components.
			full (bool=True): Do we find *all* the homological persistence events,
				or just the target one?
			stop (function): A function that returns the number of essential cycles
				found before sampling the next configuration.
			tries (int=64): Number of persistence computation attempts made before
				re-sampling.

		<center> <button type="button" class="collapsible" id="InvadedCluster-Persistence-2">Performance in \(\mathbb T^2_N\)</button> </center>
		..include:: ./tables/InvadedCluster.Persistence.2.html

		<center> <button type="button" class="collapsible" id="InvadedCluster-Persistence-4">Performance in \(\mathbb T^4_N\)</button> </center>
		..include:: ./tables/InvadedCluster.Persistence.4.html
		"""
		# Object access.
		self.complex = C
		self.dimension = dimension
		self.stop = stop
		self._returns = 3
		self.field = field

		self.tries = tries
		self.full = full

		# Check if we're debugging.
		self._DEBUG = kwargs.get("_DEBUG", False)


		# Force-recompute the matrices for a different dimension; creates
		# a set of orientations for fast elementwise products.
		self.matrices = Matrices()
		self.matrices.full = self.complex.matrices.full

		boundary, coboundary = self.complex.recomputeBoundaryMatrices(dimension)
		self.matrices.boundary = boundary
		self.matrices.coboundary = coboundary


		# Useful values to have later.
		self.cellCount = len(self.complex.flattened)
		self.cells = len(self.complex.Boundary[self.dimension])
		self.faces = len(self.complex.Boundary[self.dimension-1])
		self.target = np.arange(self.complex.breaks[self.dimension], self.complex.breaks[self.dimension+1])


		# Check the dimensions of the boundary/coboundary matrices by comparing
		# the number of cells. LinBox is really sensitive to smaller-size matrices,
		# but can easily handle large ones.
		if self.cells*self.faces < 100000:
			warnings.warn(f"complex with {self.cells*self.faces} boundary matrix entries is too small for accurate matrix solves; may segfault.", TooSmallWarning, stacklevel=2)


		# Premake the "occupied cells" array; change the dimension of the complex
		# to correspond to the provided dimension.
		self.rank = comb(len(self.complex.corners), self.dimension)


		# Delegates computation for persistence and cocycle sampling.
		self._delegateComputation()


		# Seed the random number generator.
		self.RNG = np.random.default_rng()


		# If no initial spin configuration is passed, initialize.
		if not initial: self.spins = self._initial()
		else: self.spins = (initial%self.field).astype(FINT)

	
	def _delegateComputation(self):
		low, high = self.complex.breaks[self.dimension], self.complex.breaks[self.dimension+1]

		# If we're using LinBox, our sampling method is Lanczos regardless of
		# dimension.
		def sample(zeros):
			try:
				return np.array(ReducedKernelSample(
					self.matrices.coboundary, zeros, 2*self.dimension,
					self.faces, self.field, self._DEBUG
				), dtype=np.int8)
			except Exception as e:
				raise NumericalInstabilityWarning(e)
	
		# If we're using LinBox and the characteristic of our field is greater
		# than 2, we use the twist_reduce variant implemented in this library.
		if self.field > 2:
			Twister = Twist(self.field, self.matrices.full, self.complex.breaks, self.cellCount, self.dimension, self._DEBUG)
			Twister.LinearComputeCobasis();

			def _user_continue_prompt():
				_affirm = {"", "yes", "y", "ye"}
				_cont = input("\ncontinue? [y/n, default y] --> ")

				if _cont not in _affirm: sys.exit(1)
				print("\n\n")

			def persist(filtration):
				# Set defaults.
				essential = set()
				tries = 0

				# Debugging mode. Compares the SparseRREF/SpaSM output to the
				# classical persistence algorithm (LinearComputePercolationEvents).
				# NOTE: this waits for user input to continue after each iteration.
				if self._DEBUG:
					print("######################################")
					print("#### SPARSERREF/SPASM COMPUTATION ####")
					print("######################################")

					rstart = time.time()
					att = 0

					while len(essential) < self.rank:
						stop = 0 if self.full else self._STOP
						essential = Twister.RankComputePercolationEvents(filtration, stop=stop)
						att += 1

						if not self.full and len(essential) == 1: break;
					
					rend = time.time()
					essential = np.array(list(essential))
					essential = essential[(essential >= low) & (essential < high)]
					essential.sort()

					print("\n\n")
					print(essential)
					print(f"time: {(rend-rstart):.4f} :: {att} attempts")
					_user_continue_prompt()


					# print("######################################")
					# print("####      LINEAR COMPUTATION      ####")
					# print("######################################")
					# lstart = time.time()
					# _essential = Twister.LinearComputePercolationEvents(filtration)
					# lend = time.time()
					# _essential = np.array(list(_essential))
					# _essential = _essential[(_essential >= low) & (_essential < high)]
					# _essential.sort()

					# print("\n\n")
					# print(_essential, f"time: {(lend-lstart):.4f}")
					# _user_continue_prompt()

				else:
					# Attempt the persistence computation up to `self.tries` times;
					# if we exceed the attempts budget, re-sample the filtration and
					# try again.
					while len(essential) < self.rank:
						# Compute essential cycle birth times.
						stop = 0 if self.full else self._STOP
						essential = Twister.RankComputePercolationEvents(filtration, stop=stop)
						tries += 1

						# If we don't have enough essential cycles and we've tried
						# too many times, re-sample the filtration and try again.
						if len(essential) < self.rank and tries >= self.tries:
							print(f"[Persistence] exceeded the acceptable number of {self.tries} attempts. Re-sampling filtration...")
							filtration = self._filtrate(self.spins)
							tries = 0

						if not self.full and len(essential) == 1: break

					essential = np.array(list(essential))
					essential = essential[(essential >= low) & (essential < high)]
					essential.sort()
				
				return essential
		
		# If we're using LinBox and the field we're computing over *is* two,
		# use PHAT.
		elif self.field < 3:
			times = set(range(self.cellCount))

			def whittle(pairs):
				_births, _deaths = zip(*pairs)
				births = set(_births)
				deaths = set(_deaths)
				_essential = sorted(set(
					e for e in times-(births|deaths)
					if low <= e < high
				))

				if self.full: return np.array(_essential, dtype=int)
				else: return np.array([_essential[self._STOP]], dtype=int)
			
			
			def persist(filtration):
				essential = ComputePersistencePairs(self.matrices.full, filtration, self.dimension, self.complex.breaks)
				return whittle(essential)

		self.sample = sample
		self.persist = persist


	def _filtrate(self, cochain):
		"""
		Constructs a filtration based on the evaluation of the cochain.
		"""
		# Find which cubes get zeroed out (i.e. are sent to zero by the cocycle).
		boundary = self.complex.Boundary[self.dimension]
		q = self.field
		
		coefficients = cochain[boundary]
		coefficients[:,1::2] = -coefficients[:,1::2]%q
		sums = coefficients.sum(axis=1)%q

		# POSSIBLY INEFFICIENT!! TAKE A LOOK DUMMY
		satisfied = np.nonzero(sums==0)[0]
		unsatisfied = np.nonzero(sums>0)[0]
		m = satisfied.shape[0]

		# Construct the filtration.
		filtration = np.arange(self.cellCount)
		low = self.complex.breaks[self.dimension]
		high = self.complex.breaks[self.dimension+1]

		shuffled = np.random.permutation(satisfied)
		filtration[low:low+m] = self.target[shuffled]
		filtration[low+m:high] = self.target[unsatisfied]

		return filtration, shuffled, satisfied


	def _initial(self):
		"""
		Computes an initial state for the model's Complex.

		Returns:
			A numpy `np.array` representing a vector of spin assignments.
		"""
		return self.RNG.integers(
			0, high=self.field, dtype=FINT, size=self.faces
		)
	

	def proposal(self, time):
		"""
		Proposal scheme for generalized invaded-cluster evolution on the
		random-cluster model.

		Args:
			time (int): Step in the chain.

		Returns:
			A 3-tuple:

			1. a NumPy array of spin assignments;
			2. a boolean array where each column corresponds to a \(d\)-cell,
				and each row corresponds to a \(d\)-dimensional homological
				percolation event, so each entry indicates the presence or
				absence of that \(d\)-cell when the \(d\)-dimensional homological
				percolation event occurred;
			3. a boolean array where each column correponds to a \(d\)-cell,
				and each entry indicates whether its corresponding \(d\)-cell
				was satisfied under the given spin assignment.
		"""
		# Set a stopping time.
		self._STOP = self.stop();

		# Construct the filtration and find the essential cycles.
		filtration, shuffledIndices, satisfiedIndices = self._filtrate(self.spins)
		essential = self.persist(filtration)

		j = 0
		low = self.complex.breaks[self.dimension]
		satisfied = np.zeros(self.cells, dtype=int)

		# If we only computed a desired percolation time, then the matrix of
		# occupied cell indices has one row...
		if not self.full:
			t = essential[0]

			occupied = np.zeros(self.cells, dtype=int)
			occupiedIndices = shuffledIndices[:t-low+1]
			occupied[occupiedIndices] = 1

			spins = self.sample(occupiedIndices)
		# Otherwise, it has `rank` rows.
		else:
			occupied = np.zeros((self.rank, self.cells), dtype=int)

			for t in sorted(essential):
				occupiedIndices = shuffledIndices[:t-low+1]
				occupied[j,occupiedIndices] = 1

				if (j+1) == self._STOP: spins = self.sample(occupiedIndices)

				j += 1
		
		satisfied[satisfiedIndices] = 1

		return spins, occupied, satisfied
	

	def _assign(self, cocycle):
		"""
		Updates mappings from faces to spins and cubes to occupations.

		Args:
			cocycle (np.array): Cocycle on the subcomplex.
		
		Returns:
			Nothing.
		"""
		self.spins = cocycle

