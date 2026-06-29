

# distutils: language=c++

from ..common cimport (
	INDEXFLAT, Vectorize, INDEXTYPE, DATATYPE, BoundaryMatrix, Row, Column, Map,
	Basis, Bases, printBoundaryMatrix, bool
)

from .Persistence cimport (
	LinearComputePercolationEvents, LinearComputeBases, RankComputePercolationEvents,
	PHATComputePersistencePairs as _PHATComputePersistencePairs, ComputeCobasis,
	SolveComputePercolationEvents, SRankComputePercolationEvents
)

from cython.operator cimport dereference, postincrement
from libc.math cimport pow


cdef class Twist:

	def __init__(
			self,
			char characteristic,
			INDEXFLAT boundary,
			INDEXFLAT breaks,
			int cellCount,
			int dimension=-1,
			bool __DEBUG=False
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
			__DEBUG (bool): Do we want to run in debug mode? This shows standard
				error stream output from the matrix reducers.

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
		essential = Twist.LinearComputePercolationEvents(filtration)
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

		cdef int N = self.breaks[self.dimension]-self.breaks[self.dimension-1];
		self.partialBoundary = self.PartialBoundaryMatrix(self.dimension);
		self.partialCoboundary = self.__transpose(self.partialBoundary, N);

		# Construct arithmetic operations.
		self.__arithmetic();

		# Debugging?
		self.__DEBUG = __DEBUG;

		# Construct bases. This will take a LONG time sometimes, but the
		# precomputation should be worth it.
		# TODO maybe we can pass pre-computed bases as an argument? That would
		# be super slick.
		self.bases = Bases();
		self.cobasis = self.LinearComputeCobasis();

		# Construct an augmented matrix where the first <whatever> columns are
		# the cobasis, and the rest is the coboundary matrix.
		cdef int i = 0, bSize = self.cobasis.size(), pBSize = self.partialCoboundary.size();
		self.augmentedCoboundary = BoundaryMatrix(bSize+pBSize, Column());

		for i in range(bSize+pBSize):
			if i < bSize: self.augmentedCoboundary[i] = Column(self.cobasis[i]);
			else: self.augmentedCoboundary[i] = Column(self.partialCoboundary[i-bSize]);


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


	cdef BoundaryMatrix __transpose(self, BoundaryMatrix A, int columns) noexcept:
		"""
		Finds the transpose of the given BoundaryMatrix.
		"""
		cdef BoundaryMatrix T = BoundaryMatrix(columns, Column());
		cdef INDEXTYPE col, row;
		cdef DATATYPE q;
		cdef Column column;
		cdef Column.iterator it;

		for col in range(A.size()):
			column = A[col];
			it = column.begin();

			while it != column.end():
				row = dereference(it).first;
				q = dereference(it).second;
				T[row][col] = q;

				postincrement(it);

		return T;


	cdef BoundaryMatrix __asRows(self, BoundaryMatrix A, int rows) noexcept:
		cdef BoundaryMatrix R = BoundaryMatrix(rows, Row());
		cdef int row, col;
		cdef DATATYPE q;
		cdef Column column;
		cdef Column.iterator it;

		for col in range(A.size()):
			column = A[col];
			it = column.begin();

			while it != column.end():
				row = dereference(it).first;
				q = dereference(it).second;
				R[row][col] = q;
				postincrement(it);
		
		return R;


	cdef BoundaryMatrix __asColumns(self, BoundaryMatrix A, int columns) noexcept:
		cdef BoundaryMatrix C = BoundaryMatrix(columns, Column());
		cdef int r, c;
		cdef DATATYPE q;
		cdef Row row;
		cdef Row.iterator it;

		for r in range(A.size()):
			row = A[r];
			it = row.begin();

			while it != row.end():
				c = dereference(it).first;
				q = dereference(it).second;
				C[c][r] = q;
				postincrement(it);

		return C;

	
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


	cdef BoundaryMatrix ReindexPartialBoundaryMatrix(self, BoundaryMatrix boundary, INDEXFLAT filtration) noexcept:
		cdef BoundaryMatrix reindexed = BoundaryMatrix();
		cdef Index partial = self.ReindexPartialFiltration(filtration);
		cdef int t;

		for t in range(boundary.size()): reindexed.push_back(boundary[partial[t]]);

		return reindexed;


	cdef BoundaryMatrix PartialBoundaryMatrix(self, int dimension) noexcept:
		"""
		Fills the partial boundary matrix of the appropriate dimension.
		"""
		cdef int lower, low, high, col, row;
		cdef BoundaryMatrix boundary;
		cdef Column existing, relabeled;

		# Normalize indices.
		lower = self.breaks[dimension-1];
		low = self.breaks[dimension];
		high = self.breaks[dimension+1] if dimension+1 < self.breaks.size() else self.cellCount;

		boundary = BoundaryMatrix(high-low);

		for col in range(low, high):
			existing = self.referenceBoundary[col];
			relabeled = Column();

			# This is clunky.
			for existingRow, existingCoefficient in existing:
				relabeled[<INDEXTYPE>(existingRow-lower)] = existingCoefficient;

			boundary[col-low] = relabeled;
		
		return boundary;


	cdef Index ReindexPartialFiltration(self, INDEXFLAT filtration) noexcept:
		cdef int low, high, t;
		cdef Index reindexed;
		low = self.breaks[self.dimension];
		high = self.breaks[self.dimension+1];

		reindexed = Index(high-low);

		for t in range(low, high):
			reindexed[t-low] = filtration[t]-low;

		return reindexed;


	cpdef Set LinearComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		**NOTE**: `Twist.RankComputePercolationEvents()` is recommended for
		general use.

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


	cpdef Set RankComputePercolationEvents(self, INDEXFLAT filtration, int stop=0) noexcept:
		"""
		Given a filtration --- i.e. a reordering of the columns of the full
		boundary matrix --- gives times at which giant cycles of dimension
		`dimension` were created.

		Args:
			filtration (np.ndarray): An array of column indices.
			stop (int=0): The rank at which we stop searching. For example, if
				we're computing persistence on a 4-torus and `stop` is 3, then
				we find only the time at which the third giant cycle was born.
				If `stop` is 0 (or falsy), then find all the birth times.

		Returns:
			A set containing indices at which essential cycles appear.
		"""
		cdef Basis cobasis;

		# Check whether we've already computed the cobasis. If we haven't, do so!
		if self.cobasis.size() < 1:
			print("[Persistence] computing cobasis... ", end="")
			cobasis = self.LinearComputeCobasis();
			print("done.")
		else: cobasis = self.cobasis;

		# If we haven't encountered a giant cocycle, then each element fi of the
		# cobasis must be in the image of the (d-1)th coboundary matrix (which
		# sends (d-1)-cochains to d-cochains), so fi is a cocycle *and*
		# a coboundary --- which means we haven't encountered a giant cocycle yet.
		# Once fi isn't in the image of the (d-1)th coboundary matrix (i.e. it's
		# not a coboundary), then we've encountered a giant cycle, and we've
		# percolated. In this variant, we're going to prepend the cobasis to the
		# front of the (d-1)th coboundary matrix (i.e. stacking it on top of the
		# dth boundary matrix) and compute the ranks instead of solving; this
		# might save some time. Profiling will tell.
		cdef BoundaryMatrix combined, combinedT, rcombinedT, coboundary;
		cdef int c, M = self.breaks[self.dimension]-self.breaks[self.dimension-1];
		cdef Set _events, events;
		cdef Index partial;

		coboundary = self.partialCoboundary;
		combined = BoundaryMatrix();

		for c in range(cobasis.size()): combined.push_back(cobasis[c]);
		for c in range(coboundary.size()): combined.push_back(coboundary[c]);

		combinedT = self.__transpose(combined, self.partialBoundary.size());
		rcombinedT = self.ReindexPartialBoundaryMatrix(combinedT, filtration);

		# First condition computes all the percolation times; second stops after
		# the desired one is found.
		if not stop:
			_events = SRankComputePercolationEvents(
				rcombinedT, cobasis.size()+coboundary.size(), rcombinedT.size(),
				cobasis.size(), self.characteristic, self.__DEBUG
			);
		else:
			_events = SRankComputePercolationEvents(
				rcombinedT, cobasis.size()+coboundary.size(), rcombinedT.size(),
				cobasis.size(), self.characteristic, stop, self.__DEBUG
			);
		
		events = Set();

		cdef Set.iterator sit = _events.begin();
		while sit != _events.end():
			events.insert(dereference(sit)+self.breaks[self.dimension]);
			postincrement(sit);

		return events;


	cpdef Basis LinearComputeBasis(self) noexcept:
		"""
		Computes bases for the `dimension`th homology group.

		Returns:
			A `Basis` (equivalently, a `BoundaryMatrix`) of sparse vectors that
			form the basis for the `dimension`th homology group.
		"""
		if self.bases.size() > 0: return self.bases[self.dimension];

		cdef Bases existingBases = LinearComputeBases(
			self.characteristic, self.flatAddition, self.flatMultiplication, self.negation, self.inversion,
			self.referenceBoundary, self.breaks, self.cellCount, self.dimension
		);

		# Reindex the bases.
		cdef Bases reindexedBases = Bases(existingBases.size());
		cdef Basis existingBasis, reindexedBasis;
		cdef Column existingColumn, reindexedColumn;
		cdef INDEXTYPE low, i, j, row, col;
		cdef DATATYPE q;

		for i in range(existingBases.size()):
			low = self.breaks[i];
			existingBasis = existingBases[i];
			reindexedBasis = Basis(existingBasis.size());

			for j in range(existingBasis.size()):
				existingColumn = existingBasis[j];
				reindexedColumn = Column();

				for row, q in existingColumn:
					reindexedColumn[row-low] = q;

				reindexedBasis[j] = reindexedColumn;

			reindexedBases[i] = reindexedBasis;

		self.bases = reindexedBases;
		return self.bases[self.dimension];

	
	cpdef Basis LinearComputeCobasis(self) noexcept:
		"""
		Computes a basis for the `dimension`th cohomology group, relative to
		the basis chosen by `Twist.LinearComputeBasis()`.

		Returns:
			A `Basis` (equivalently, a `BoundaryMatrix`) of sparse vectors that
			form the basis for the `dimension`th homology group.
		"""
		# Compute a basis for the dth homology group.
		self.LinearComputeBasis();

		# To compute the cobasis, we require that each fi(ej) = 1(i=j), and that
		# each fi is in the kernel of the dth coboundary matrix. To solve all these
		# problems at once, we "stack" the cycle basis on top of the dth coboundary
		# matrix to form a matrix B; fi is then a solution to the equation Bx=g,
		# where g is a vector of length (cycle basis rank) + (# of (d+1)-cells)
		# with a 1 in the ith position and zeros elsewhere.

		# Since our matrices are column-major, we will compute the (d+1)th 
		# boundary matrix (which is the transpose of the dth coboundary matrix),
		# append the cycle basis to the front of the boundary matrix, then take
		# the transpose.
		cdef BoundaryMatrix combined, boundary = self.PartialBoundaryMatrix(self.dimension+1);
		cdef Basis cocyclebasis, cyclebasis = self.bases[self.dimension];
		cdef int rows, rank = cyclebasis.size();

		rank = cyclebasis.size();
		rows = self.breaks[self.dimension+1]-self.breaks[self.dimension];

		combined = BoundaryMatrix();
		for c in range(rank): combined.push_back(cyclebasis[c]);
		for c in range(boundary.size()): combined.push_back(boundary[c]);

		cocyclebasis = ComputeCobasis(combined, rows, combined.size(), rank, self.characteristic, self.__DEBUG);
		self.cobasis = cocyclebasis;
		return self.cobasis;


cpdef PersistencePairs PHATComputePersistencePairs(INDEXFLAT boundary, INDEXFLAT filtration, int homology, INDEXFLAT breaks) noexcept:
	"""
	Computes the persistence pairs of the complex corresponding to the provided
	boundary matrix and filtration.

	Args:
		boundary (np.array): Flattened boundary matrix given by `Complex.matrices.full`.
		filtration (np.array): Permutation on the order of the columns of the
			boundary matrix. **Here, we assume that only cells of dimension `homology`
			are being permuted. Shuffling the order of cells of different
			dimensions will result in incorrect computations.**
		homology (int): Homology group we're interested in; corresponds to the
			dimension of permuted cells.
		breaks (np.array): Index ranges for cells by dimension, given by
			`Complex.breaks`.

	Returns:
		A list of [birth, death] pairs.
	"""
	cdef Index _boundary, _filtration, _breaks;
	_boundary = Vectorize(boundary);
	_filtration = Vectorize(filtration);
	_breaks = Vectorize(breaks);

	return _PHATComputePersistencePairs(_boundary, _filtration, homology, _breaks);



