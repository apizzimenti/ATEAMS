
# distutils: language=c++

from ..common cimport (
	INDEXFLAT, Vectorize, DATATYPE, BoundaryMatrix, Column, Map, Basis, Bases
)

from .LinBoxMethods cimport (
	ComputePercolationEvents, ZpComputePercolationEvents,
	LinearComputePercolationEvents, LinearComputeBases
)

from libc.math cimport pow


cdef class Twist:

	def __init__(
			self,
			char characteristic,
			INDEXFLAT boundary,
			INDEXFLAT breaks,
			int cellCount,
			int dimension=-1
		):
		"""
		Implements the `twist_reduce` algorithm from Chen and Kerber (2011) and
		PHAT (2017). Implemented as a class so we minimally re-compute boundary
		matrix things; we also move away from NumPy integers and toward C-style
		ones for compatibility.

		Args:
			characteristic (char): Characteristic of the finite field \(\mathbb Z/p\mathbb Z\).
			boundary (np.ndarray): The full flattened boundary matrix. Given a
				complex `C` (e.g. `ateams.complexes.Cubical`), `C.matrices.full`.
			breaks (np.ndarray): Indices at which cells of a new dimension begin
				in `boundary`. Given a complex `C` (e.g. `ateams.complexes.Cubical`), `C.breaks`.
			cellCount (int): Total number of cells in the complex. Given a model
				`M` (e.g. `ateams.models.SwendsenWang`), `M.cellCount`.
			dimension (int): Dimension of percolation. Given a model `M` (e.g.
				`ateams.models.SwendsenWang`), `M.dimension`.

		For example,

		```python
		from ateams.complexes import Cubical
		from ateams.arithmetic import Twist
		import numpy as np

		# Create a lattice; set the cell count, dimension, field characteristic.
		C = Cubical().fromCorners([3,3])
		cellCount = len(C.flattened)
		dimension = 1
		field = 3

		# Create the Twist object.
		Twister = Twist(field, C.matrices.full, C.breaks, cellCount, dimension)

		# Construct a filtration and compute the percolation events.
		filtration = np.arange(cellCount)
		essential = Twist.ComputePercolationEvents(filtration)
		```

		In this case, `essential` should be `{0, 17, 35, 21}`, and the times \(17\)
		and \(21\) are the essential \(1\)-dimensional cycles.
		"""
		self.characteristic = characteristic;
		self.cellCount = cellCount;
		self.dimension = dimension;
		self.breaks = Vectorize(breaks);
		self.topDimension = self.dimension+1 if self.dimension < self.breaks.size() else self.breaks.size();

		self.fullBoundary = Vectorize(boundary);
		self.referenceBoundary = self.FillBoundaryMatrix(boundary);
		self.workingBoundary = BoundaryMatrix(self.referenceBoundary);
		self.bases = Bases(self.breaks.size());

		# Construct arithmetic operations.
		self.__arithmetic();


	cdef void __arithmetic(self) noexcept:
		# Given a field characteristic, construct addition and multiplication
		# tables.
		cdef int p = self.characteristic;
		cdef Table addition, multiplication;
		cdef Lookup negation, inverse, flatAddition, flatMultiplication;
		cdef int i, j;

		addition = Table(p, Lookup(p));
		multiplication = Table(p, Lookup(p));

		# Addition, multiplication tables.
		for i in range(p):
			for j in range(p):
				addition[i][j] = <DATATYPE>((i+j)%p);
				flatAddition.push_back(<DATATYPE>((i+j)%p));

				multiplication[i][j] = <DATATYPE>((i*j)%p);
				flatMultiplication.push_back(<DATATYPE>((i*j)%p));

		self.addition = addition;
		self.flatAddition = flatAddition;

		self.multiplication = multiplication;
		self.flatMultiplication = flatMultiplication;

		# Negations and inverses.
		negation = Lookup(p);
		inverse = Lookup(p);

		negation[0] = 0;
		inverse[0] = 0;

		for i in range(1, p):
			negation[i] = <DATATYPE>(p-i);
			inverse[i] = <DATATYPE>(pow(i, p-2)%p);

		self.negation = negation;
		self.inversion = inverse;
	

	cdef BoundaryMatrix FillBoundaryMatrix(self, INDEXFLAT boundary) noexcept:
		"""
		Fills the boundary matrix.
		"""
		cdef BoundaryMatrix B = BoundaryMatrix(self.cellCount);
		cdef int row, column, q, t, N = boundary.shape[0];
		cdef int p = self.characteristic;

		for t in range(0, N, 3):
			row = boundary[t];
			column = boundary[t+1];
			q = boundary[t+2];

			if (q != 0): B[column][row] = <DATATYPE>((q+p)%p);

		return B;

	
	cdef BoundaryMatrix ReindexBoundaryMatrix(self, INDEXFLAT filtration) noexcept:
		cdef int low = self.breaks[self.dimension];
		cdef int high = self.breaks[self.dimension+1];
		cdef int higher = self.breaks[self.dimension+2] if self.dimension+2 < self.breaks.size()-1 else self.cellCount;
		cdef int t, L, row, column;
		cdef Column existing, reindexed;
		cdef char q;

		# Construct an index mapper.
		cdef Map IndexMap = Map();
		for t in range(low, higher): IndexMap[filtration[t]] = t;

		# Loop over the appropriate cells, moving or reindexing where necessary.
		for t in range(0, self.fullBoundary.size(), 3):
			row = self.fullBoundary[t];
			column = self.fullBoundary[t+1];

			if (low <= column < high):
				# In this situation, we're editing the column --- for example,
				# if the (usual) 9th element was placed 10th, then we have to
				# change the 9th column to the 10th one.
				self.workingBoundary[column] = self.referenceBoundary[filtration[column]];
			elif (high <= column < higher):
				# Otherwise, edit the row. For some reason, editing it in this way
				# makes a difference.
				reindexed = Column();
				existing = self.referenceBoundary[column];

				for it in existing:
					row = it.first;
					q = it.second;
					reindexed[IndexMap[row]] = q;
				
				self.workingBoundary[column] = reindexed;
		
		return self.workingBoundary;

	
	cpdef Set ComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		Given a filtration --- i.e. a reordering of the columns of the full
		boundary matrix --- gives times at which essential cycles of dimension
		`dimension` were created. Performs arithmetic using addition and
		multiplication tables stored in `vector<vector<char>>`s.

		Args:
			filtration (np.ndarray): An array of column indices.

		Returns:
			A set containing indices at which essential cycles appear.
		"""
		self.workingBoundary = self.ReindexBoundaryMatrix(filtration);

		return ComputePercolationEvents(
			self.addition, self.multiplication, self.negation, self.inversion,
			self.workingBoundary, self.breaks, self.cellCount
		);


	cpdef Set LinearComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		Given a filtration --- i.e. a reordering of the columns of the full
		boundary matrix --- gives times at which essential cycles of dimension
		`dimension` were created. Performs arithmetic using flattened addition
		and multiplication tables stored in `vector<char>`s.

		Args:
			filtration (np.ndarray): An array of column indices.

		Returns:
			A set containing indices at which essential cycles appear.
		"""
		self.workingBoundary = self.ReindexBoundaryMatrix(filtration);

		return LinearComputePercolationEvents(
			self.characteristic, self.flatAddition, self.flatMultiplication, self.negation, self.inversion,
			self.workingBoundary, self.breaks, self.cellCount, self.dimension
		);
	

	cpdef Set ZpComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		Given a filtration --- i.e. a reordering of the columns of the full
		boundary matrix --- gives times at which essential cycles of dimension
		`dimension` were created. Performs arithmetic using `Givaro`.

		Args:
			filtration (np.ndarray): An array of column indices.

		Returns:
			A set containing indices at which essential cycles appear.
		"""
		self.workingBoundary = self.ReindexBoundaryMatrix(filtration);

		return ZpComputePercolationEvents(
			self.characteristic, self.workingBoundary, self.breaks, self.cellCount
		);


	cpdef Bases LinearComputeBases(self) noexcept:
		"""
		Computes bases for the homology groups of the underlying space.

		Returns:
			A `Bases` object (which is just a `std::vector` of `Basis` objects)
			for the underlying space. Given the configuration of `LinearComputeBases`,
			only bases for the top-level dimension down to `dimension-1` are given.
		"""
		self.bases = LinearComputeBases(
			self.characteristic, self.flatAddition, self.flatMultiplication, self.negation, self.inversion,
			self.referenceBoundary, self.breaks, self.cellCount, self.dimension
		);

		return self.bases;

