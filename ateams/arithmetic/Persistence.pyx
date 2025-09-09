
# distutils: language=c++

from ..common cimport (
	INDEXFLAT, FFINT, TABLECONTIG, FLATCONTIG, Index, Set, Column, BoundaryMatrix,
	FlatBoundaryMatrix, MatrixEntries, SparseLinearCombination, bool
)
from ..common import FINT

import numpy as np
cimport numpy as np

from cython.operator cimport dereference
from libc.math cimport pow


cdef class Persistence:
	"""
	Computes the persistent homology of a complex with coefficients
	in an arbitrary finite field.
	"""
	def __init__(
			self,
			FFINT characteristic,
			list[list[int]] flattened,
			int homology=-1
		):
		r"""
		Args:
			characteristic (int): Characteristic of the finite field over which
				we do computations.
			flattened (list): List-of-lists representing the *original* flattened
				boundary matrix for the complex.
			homology (int=-1): Degree of homology observed for homological percolation events;
				if not specified, the homology of the codimension-1 skeleton is
				reported.

		The `Persistence.ComputePercolationEvents` method computes the birth
		times of giant cycles given a complete filtration — that is, a filtration
		where the terminal element is the entire cubical complex. For example,
		the code below finds the birth times of giant cycles on the default
		\(3 \times 3\) cubical torus: the filtration stored in `filtration`
		adds each cube to the cubical complex in order of construction. At
		completion, `events` contains the times `{17, 21}` which correspond to
		crossings of the meridian and equator of the torus, respectively.
		
		
			from ateams.arithmetic import Persistence
			from ateams.structures import Complex
			
			L = Complex().fromCorners([3,3], field=3)
			P = Persistence(L.field.characteristic, L.flattened, homology=1)

			filtration = np.arange(len(L.flattened))
			events = P.ComputePercolationEvents(filtration)


		The `Persistence.ComputeBettiNumbers` method provides functionality to compute the
		Betti numbers for an entire subcomplex. The subcomplex is specified by
		a list of indices corresponding to the cells included in the subcomplex.
		Initializing the `Persistence` object as before, we can get the Betti
		numbers of the \(2\)-torus encoded by the `Complex` above by passing a
		list of all cells' indices in the flattened boundary matrix: since there
		are \(36\) cells (including vertices, edges, and squares), the `subcomplex`
		is just the range of integers \(0-36\):


			from ateams.arithmetic import Persistence
			from ateams.structures import Complex
			
			L = Complex().fromCorners([3,3], field=3)
			P = Persistence(L.field.characteristic, L.flattened)

			subcomplex = np.arange(len(L.flattened))

			bettis = P.ComputeBettiNumbers(subcomplex)


		The result in `bettis` is the list `[1,2,1]`. By deleting an edge, we
		reduce the number of squares by two, so we should expect the homology
		of a one-edge-deleted \(2\)-torus to have betti numbers `[1,2,0]`. To
		access the indices of the flattened boundary matrix corresponding to
		cells of each dimension, we can use the `Complex.tranches` property
		which maps an integer dimension to a (start, stop) pair: for example,
		`L.tranches[1]` is `[9, 27]` in the complex above, so we know that all
		entries in the flattened boundary matrix between indices \(9\) and \(27\)
		(_right-exclusive_) correspond to edges. To test our theory, we
		construct a subcomplex by deleting the first edge added:


			from ateams.arithmetic import Persistence
			from ateams.structures import Complex
			
			L = Complex().fromCorners([3,3], field=3)
			P = Persistence(L.field.characteristic, L.flattened)

			subcomplex = np.arange(len(L.flattened))
			firstEdge = L.tranches[1][0]
			subcomplex = np.concatenate([subcomplex[:firstEdge], subcomplex[firstEdge+1:]])

			bettis = P.ComputeBettiNumbers(subcomplex)


		The result in `bettis` is the list `[1,2,0]`, as expected. Note that we
		only need to pass a `homology` parameter to the `Persistence` constructor
		if we call `Persistence.ComputePercolationEvents`.
		"""
		self.characteristic = characteristic;
		self.boundary = self.Vectorize(flattened);
		
		# If there was no homology parameter passed, just compute the homology
		# of the codimension-1 skeleton.
		if homology < 0:
			# Knock off *two* dimensions (so we don't get off-by-one errors), and
			# make sure we're computing at least the zeroth homology.
			self.homology = self._tranches.size()-2;
			if self.homology < 0: self.homology = 0;
		else: self.homology = homology;

		self.cellCount = self._boundary.size();
		self.vertexCount = self._tranches[0][1];
		self.higherCellCount = self._tranches[self.homology+1][1];
		self.low = self._tranches[self.homology][0];
		self.high = self._tranches[self.homology][1];

		# Construct arithmetic operations.
		self.__arithmetic();

		# Load premarked indices into a `Vector` so we can refresh quickly.
		cdef int i;

		self.premarked = Index(self.vertexCount);
		for i in range(self.vertexCount): self.premarked[i] = i;

		# Data structures for storing column information.
		self.columnEntries = MatrixEntries(self.cellCount);
		self.columnEntriesIterable = FlatBoundaryMatrix(self.cellCount);
		self.columnEntriesCoefficients = BoundaryMatrix(self.cellCount);
		self.linearCombinations = BoundaryMatrix(self.cellCount);

		self.__flushDataStructures();


	cdef void __arithmetic(self) noexcept:
		# Given a field characteristic, construct addition and multiplication
		# tables.
		cdef int p = self.characteristic;
		cdef TABLECONTIG addition, multiplication;
		cdef FLATCONTIG negation, inverse;
		cdef int i, j;

		addition = np.zeros((p,p), dtype=FINT);
		multiplication = np.zeros((p,p), dtype=FINT);

		# Addition, multiplication tables.
		for i in range(p):
			for j in range(p):
				addition[i,j] = <FFINT>((i+j)%p);
				multiplication[i,j] = <FFINT>((i*j)%p);

		self.addition = addition;
		self.multiplication = multiplication;

		# Negations and inverses.
		negation = np.zeros(p, dtype=FINT);
		inverse = np.zeros(p, dtype=FINT);

		negation[0] = 0;
		inverse[0] = 0;

		for i in range(1, p):
			negation[i] = <FFINT>(p-i);
			inverse[i] = <FFINT>(pow(i, p-2)%p);

		self.negation = negation;
		self.inverse = inverse;

	
	cdef void __flushDataStructures(self, bool premark=True) noexcept:
		# Data structures for storing column information.
		self.columnEntries = MatrixEntries(self.cellCount);
		self.columnEntriesIterable = FlatBoundaryMatrix(self.cellCount);
		self.columnEntriesCoefficients = BoundaryMatrix(self.cellCount);
		self.linearCombinations = BoundaryMatrix(self.cellCount);

		self.marked = Set();
		if premark: self.marked.insert(self.premarked.begin(), self.premarked.end());
		self.markedIterable = Index(self.cellCount);
		self.nextColumnAdded = Index(self.cellCount, 0);


	cdef FlatBoundaryMatrix ReorderBoundary(self, INDEXFLAT filtration) noexcept:
		"""
		Re-indexes the boundary matrix according to the given filtration.

		Args:
			filtration (np.array): `NumPy` array specifying the order in which
				cells from the complex are added.

		Returns:
			A C++ `std::vector` (cast as a `NumPy` array) containing the reindexed
			boundary matrix.
		"""
		cdef int t, i, j, filtered, unfiltered, face, N, M, dimension, start, stop, please;
		cdef Index faces, indices;
		cdef FlatBoundaryMatrix reordered;

		N = self._boundary.size();
		indices = Index(N);

		# Maps the elements of the filtration to their new indices.
		for t in range(N):
			filtered = filtration[t];
			indices[filtered] = t;

		# Re-orders the (flattened) boundary matrix.
		reordered = FlatBoundaryMatrix(N);
		for t in range(N):
			reordered[t] = self._boundary[filtration[t]]

		# Re-indexes the re-ordered boundary matrix.
		for t in range(N):
			M = reordered[t].size();

			for i in range(M):
				reordered[t][i] = indices[reordered[t][i]];

		self.boundary = reordered;
		return self.boundary;
		


	cpdef FlatBoundaryMatrix ReindexBoundary(self, INDEXFLAT filtration) noexcept:
		"""
		Re-indexes the boundary matrix according to the given filtration.

		Args:
			filtration (np.array): `NumPy` array specifying the order in which
				cells from the complex are added.

		Returns:
			A C++ `std::vector` (cast as a `NumPy` array) containing the reindexed
			boundary matrix.
		"""
		cdef int t, i, j, filtered, unfiltered, face, N, M, dimension, start, stop, please;
		cdef Index faces, indices, temp, high;

		N = self._boundary.size();
		indices = Index(N);

		# Maps the elements of the filtration to their new indices.
		for t in range(N):
			filtered = filtration[t];
			indices[filtered] = t;

		start = self._tranches[self.homology][0];
		stop = self._tranches[self.homology][1];
		please = self._tranches[self.homology+1][1];

		# Swaps elements when necessary, relabels elements when necessary.
		for i in range(start, please+1):
			# We're swapping elements of dimension `self.homology`:
			if i >= start and i < stop:
				filtered = filtration[i];
				self.boundary[i] = self._boundary[filtered];
			# Otherwise, we're relabeling indices.
			elif i >= stop and i < please:
				faces = self._boundary[i];
				M = faces.size();

				for j in range(M):
					unfiltered = faces[j];
					filtered = indices[unfiltered];
					self.boundary[i][j] = filtered;
		
		return self.boundary

	
	cdef FlatBoundaryMatrix ReindexSubBoundary(self, INDEXFLAT subcomplex) noexcept:
		# Create a "subboundary" matrix that maps old boundary indices to new
		# ones.
		cdef Set included;
		cdef Index renumbering, faces, retained, dimensions;
		cdef FlatBoundaryMatrix subboundary, subsubboundary;
		cdef bool retain;
		cdef int i, j, M, N, dimension;

		# Keep track of which indices are included in the subcomplex.
		N = subcomplex.shape[0];
		included = Set();
		for i in range(N): included.insert(subcomplex[i]);

		# Build the subcomplex.
		M = self._boundary.size();
		renumbering = Index(M);
		subboundary = FlatBoundaryMatrix(M);
		retained = Index();

		for i in range(M):
			# If this cell is *not* included in the subcomplex, throw it out
			# and continue.
			retain = included.contains(i);

			# Check whether all the faces belong.
			faces = self._boundary[i];
			N = faces.size();

			for j in range(N):
				# Check whether this face is included in the subcomplex. If it
				# is *not* included, we do *not* include the cell, which means
				# we have to remove it from the set of included cell indices.
				retain = retain and included.contains(faces[j]);

				if not retain:
					included.erase(i);
					break;

			# If this cell should be included, note its original index and new
			# index.
			if retain:
				retained.push_back(i);
				renumbering[i] = retained.size()-1;
		
		# Now, construct the subboundary matrix.
		M = retained.size();
		subboundary = FlatBoundaryMatrix(M);
		dimensions = Index(M);

		for i in range(M):
			subboundary[i] = self._boundary[retained[i]];
			N = subboundary[i].size();
			dimensions[i] = <int>(N/2);

			for j in range(N):
				subboundary[i][j] = renumbering[subboundary[i][j]];

		self._dimensions = dimensions;
		self.boundary = subboundary;

		# Scan over the array of dimensions, checking how many cells of each
		# dimension exist (and where their indices stop/start).
		cdef FlatBoundaryMatrix tranches = FlatBoundaryMatrix();
		cdef Index tranche;
		cdef int t, _tranche = 0;

		N = dimensions.size();

		for t in range(1, N):
			if dimensions[t] > dimensions[t-1] or t == N-1:
				tranche = Index(2);
				tranche[0] = _tranche;
				tranche[1] = t if t < N-1 else N;
				tranches.push_back(tranche);

				_tranche = t;

		self._tranches = tranches;

		# Re-count cells.
		self.cellCount = self.boundary.size();
		self.vertexCount = self._tranches[0][1];
		self.higherCellCount = self._tranches[self.homology+1][1];
		self.low = self._tranches[self.homology][0];
		self.high = self._tranches[self.homology][1];

		# Set pre-marked vertices again.
		self.premarked = Index(self.vertexCount);
		for i in range(self.vertexCount): self.premarked[i] = i;

		return self.boundary
		

	cdef FlatBoundaryMatrix Vectorize(self, list[list[int]] flattened) noexcept:
		"""
		Convert a list of lists into C++ vectors.

		Args:
			flattened (list): Sparse boundary matrix.

		Returns:
			C++ `std::vector` (cast as as `NumPy` array) representing the same data.
		"""
		cdef FlatBoundaryMatrix outer;
		cdef Index inner, dimensions;
		cdef int i, j, lower, M, N, dimension;

		# We only want to get boundary matrices for indices of dimension 1 or
		# greater; i.e. we're excluding vertices.
		M = len(flattened);
		outer = FlatBoundaryMatrix(M);
		dimensions = Index(M);

		for i in range(M):
			# If we're dealing with a non-vertex, we just copy the original
			# boundary matrix over; otherwise, we're dealing with a vertex, which
			# has no boundary.
			try:
				N = len(flattened[i]);
				inner = Index(N);
				for j in range(N): inner[j] = flattened[i][j];

				dimension = <int>(N/2);
			except:
				inner = Index(0);
				dimension = 0;
			
			dimensions[i] = dimension;
			outer[i] = inner;

		# Set this as the current boundary matrix, but also return it (for
		# completion's sake).
		self._boundary = outer;
		self._dimensions = dimensions;

		# Scan over the array of dimensions, checking how many cells of each
		# dimension exist (and where their indices stop/start).
		cdef FlatBoundaryMatrix tranches = FlatBoundaryMatrix();
		cdef Index tranche;
		cdef int t, _tranche = 0;

		N = dimensions.size();

		for t in range(1, N):
			if dimensions[t] > dimensions[t-1] or t == N-1:
				tranche = Index(2);
				tranche[0] = _tranche;
				tranche[1] = t if t < N-1 else N;
				tranches.push_back(tranche);

				_tranche = t;

		self._tranches = tranches;
		
		return outer;


	cdef int youngestOf(self, Set column) noexcept:
		return dereference(column.rbegin());

	
	cdef Set Eliminate(self, int youngest, Set faces, Column &faceCoefficients) noexcept:
		"""
		Performs Gaussian elimination on the row specified by `faceCoefficients`
		and the row with a pivot in column `youngest`.

		.. WARNING: Cannot be called from Python; internal only.

		Args:
			youngest (int): Pivot column.
			faces (std::set): Ordered set of faces.
			faceCoefficients (&std::unordered_map[int,FFINT]): Unordered map taking
				indices to coefficients; this represents a row in the matrix.

		Returns:
			`std::set` of remaining faces.
		"""
		cdef Index entriesIterable;
		cdef int i, entry, N;
		cdef FFINT _q, inverse, q, entryCoefficient, faceCoefficient, mul, add;

		# Get the coefficient of the pivot entry, then eliminate the row.
		_q = self.columnEntriesCoefficients[youngest][youngest];
		inverse = self.inverse[_q];

		entriesIterable = self.columnEntriesIterable[youngest];
		N = entriesIterable.size();

		# Eliminate.
		for i in range(N):
			# `entry` is the index of the row (column?) we're trying to eliminate.
			entry = entriesIterable[i];
			entryCoefficient = self.columnEntriesCoefficients[youngest][entry];
			mul = self.multiplication[inverse,entryCoefficient];
			q = self.negation[mul];

			# If `add` is 0, we don't care about tracking it anymore; throw it
			# out of the faces (and remove it from the dict, if it were in there
			# in the first place?).
			if faces.contains(entry):
				faceCoefficient = faceCoefficients[entry];
				add = self.addition[faceCoefficient,q];
			else:
				add = q;

			if add < 1:
				faces.erase(entry);
				faceCoefficients.erase(entry);
			else:
				faces.insert(entry);
				faceCoefficients[entry] = add;
			
		return faces


	cdef Set RemoveUnmarkedCells(self, int cell, Set faces, Column &faceCoefficients) noexcept:
		"""
		Given the latest cell added to the filtration, remove unmarked
		(nonpivot) entries from its boundary.

		Args:
			int (cell): The most recent cell added to the filtration.

		Returns:
			An `OrderedSet` of `int`s corresponding to (indices of!) pivot
			entries in the boundary of cell `cell`.
		"""
		cdef int i, face, parity, N;
		N = self.boundary[cell].size();

		for i in range(N):
			face = self.boundary[cell][i];

			# If the face is unmarked, throw it out.
			if self.marked.contains(face):
				faces.insert(face);

				if (i%2) < 1: faceCoefficients[face] = self.negation[1];
				else: faceCoefficients[face] = self.negation[self.characteristic-1];

		return faces;

	
	cdef Set ReducePivotRow(self, int cell, Set faces, Column &faceCoefficients) noexcept:
		"""
		Reduces the pivot row corresponding to cell `cell`.

		Args:
			cell (int): The most recent cell added to the filtration.

		Returns:
			An ordered set of indices corresponding to nonzero entries in the
			of the pivot column.
		"""
		cdef Set youngestColumnEntries;
		cdef int youngest;
		cdef FFINT _q, q;

		faces = self.RemoveUnmarkedCells(cell, faces, faceCoefficients);

		while not faces.empty():
			# Get the face of `cell` of maximum degree (i.e. the one added latest
			# to the complex).
			youngest = self.youngestOf(faces);
			youngestColumnEntries = self.columnEntries[youngest];

			# If the column's empty, we've found a pivot, and we're done.
			if youngestColumnEntries.empty(): break;

			# Otherwise, eliminate.
			faces = self.Eliminate(youngest, faces, faceCoefficients);
		
		return faces;
			

	cpdef Set ComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		Computes the times of homological percolation events given a filtration
		and a boundary matrix.

		Args:
			filtration (np.array): Order in which cells are added.

		Returns:
			A `set` of times at which homological percolation occurs.
		"""
		# Flush the set of marked indices, adding premarked ones.
		self.__flushDataStructures(premark=False);
		cdef Set events = Set();

		# Construct the boundary matrix for this filtration; variables for
		# objects.
		cdef FlatBoundaryMatrix boundary = self.ReindexBoundary(filtration);
		cdef Index facesIterable, degree = Index(self.cellCount);
		cdef Set faces;
		cdef Column faceCoefficients;
		cdef int t, j, cell, dimension, time, tagged, youngest;

		# TODO: shouldn't have to iterate over vertices
		tagged = 0;

		for t in range(0, self.cellCount):
			# Since our filtrations are discrete (i.e. we add exactly one simplex
			# at each time-step), the "degree" of the cell is the same as the time
			# at which it was added.
			cell = t;

			# Create buckets for indices and coefficients; these are created and
			# stored ONCE, but accessed many times.
			faces = Set();
			faceCoefficients = Column();

			# Eliminate.
			faces = self.ReducePivotRow(cell, faces, faceCoefficients);
			
			# If we end up with no faces, we've found a pivot row.
			if faces.empty():
				# Mark the face, but only add it to the marked iterable if it's
				# of the right dimension.
				self.marked.insert(cell);
				degree[cell] = t;
				
				# if self._dimensions[cell] == self.homology:
				self.markedIterable[tagged] = cell;
				tagged = tagged + 1;

			# Otherwise, store the row in the appropriate locations.
			else:
				youngest = self.youngestOf(faces);
				facesIterable = Index();
				facesIterable.insert(facesIterable.begin(), faces.begin(), faces.end());

				self.columnEntries[youngest] = faces;
				self.columnEntriesIterable[youngest] = facesIterable;
				self.columnEntriesCoefficients[youngest] = faceCoefficients;
				degree[youngest] = t;

		# # Once we're done eliminating, check over what we find.
		cdef Set unmarked;

		for j in range(tagged):
			cell = self.markedIterable[j];

			unmarked = self.columnEntries[cell];
			if unmarked.empty(): events.insert(degree[cell]);
		
		return events


	cdef Set TwistEliminate(self, int youngest, Set &faces, Column &faceCoefficients, Column &columnsReduced) noexcept:
		"""
		Performs Gaussian elimination on the row specified by `faceCoefficients`
		and the row with a pivot in column `youngest`.

		.. WARNING: Cannot be called from Python; internal only.

		Args:
			youngest (int): Pivot column.
			faces (std::set): Ordered set of faces.
			faceCoefficients (&std::unordered_map[int,FFINT]): Unordered map taking
				indices to coefficients; this represents a row in the matrix.

		Returns:
			`std::set` of remaining faces.
		"""
		cdef Index entriesIterable;
		cdef int i, entry, N, nextAdded;
		cdef FFINT _q, inverse, q, entryCoefficient, faceCoefficient, mul, add;

		nextAdded = self.nextColumnAdded[youngest];

		# Get the coefficient of the pivot entry, then eliminate the row.
		_q = self.columnEntriesCoefficients[nextAdded][youngest];
		inverse = self.inverse[_q];
		columnsReduced[nextAdded] = self.negation[inverse];

		entriesIterable = self.columnEntriesIterable[nextAdded];
		N = entriesIterable.size();

		# Eliminate.
		for i in range(N):
			# `entry` is the index of the row (column?) we're trying to eliminate.
			entry = entriesIterable[i];
			entryCoefficient = self.columnEntriesCoefficients[nextAdded][entry];
			mul = self.multiplication[inverse,entryCoefficient];
			q = self.negation[mul];

			# If `add` is 0, we don't care about tracking it anymore; throw it
			# out of the faces (and remove it from the dict, if it were in there
			# in the first place?).
			if faces.contains(entry):
				faceCoefficient = faceCoefficients[entry];
				add = self.addition[faceCoefficient,q];
			else:
				add = q;

			if add < 1:
				faces.erase(entry);
				faceCoefficients.erase(entry);
			else:
				faces.insert(entry);
				faceCoefficients[entry] = add;
			
		return faces


	cdef Set TwistBuildFace(self, int cell, Set &faces, Column &faceCoefficients) noexcept:
		"""
		Build the face. Here, we don't delete "unmarked" cells, since the column
		deletion does that for us.

		Args:
			int (cell): The most recent cell added to the filtration.

		Returns:
			An `OrderedSet` of `int`s corresponding to (indices of!) pivot
			entries in the boundary of cell `cell`.
		"""
		cdef int i, face, parity, N;
		N = self.boundary[cell].size();

		for i in range(N):
			face = self.boundary[cell][i];
			faces.insert(face);

			if (i%2) < 1: faceCoefficients[face] = self.negation[1];
			else: faceCoefficients[face] = self.negation[self.characteristic-1];
		
		return faces;
	

	cdef Set TwistReducePivotRow(self, int cell, Set &faces, Column &faceCoefficients) noexcept:
		"""
		Reduces the pivot row corresponding to cell `cell`.

		Args:
			cell (int): The most recent cell added to the filtration.

		Returns:
			An ordered set of indices corresponding to nonzero entries in the
			of the pivot column.
		"""
		cdef Set youngestColumnEntries;
		cdef Column columnsReduced = Column();
		cdef int youngest;
		cdef FFINT _q, q;

		faces = self.TwistBuildFace(cell, faces, faceCoefficients);

		# Record the linear combinations required to reduce the column (if it's
		# not a pivot column).

		while not faces.empty() and self.nextColumnAdded[self.youngestOf(faces)] != 0:
			# Get the face of `cell` of maximum degree (i.e. the one added latest
			# to the complex).
			youngest = self.youngestOf(faces);

			# Otherwise, eliminate.
			faces = self.TwistEliminate(youngest, faces, faceCoefficients, columnsReduced);
		
		self.linearCombinations[cell] = columnsReduced;

		return faces;


	cpdef Set TwistComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		Computes the times of homological percolation events given a filtration
		and a boundary matrix. Uses a modified version of the twist_reduce
		algorithm introduced by Chen and Kerber (2011) and the one used in PHAT
		(2017).

		Args:
			filtration (np.array): Order in which cells are added.

		Returns:
			A `set` of times at which homological percolation occurs.
		"""
		# Flush the set of marked indices, adding premarked ones.
		self.__flushDataStructures();
		cdef Set events = Set();

		# Construct the boundary matrix for this filtration; variables for
		# objects.
		cdef FlatBoundaryMatrix boundary = self.ReindexBoundary(filtration);
		cdef Index facesIterable, degree = Index(self.cellCount);
		cdef Set faces;
		cdef Column faceCoefficients;
		cdef int t, j, dim, cell, dimension, time, tagged, youngest;

		tagged = 0;
		dim = self.homology+1;

		# Go backwards, according to dimension.
		while (dim > self.homology-1):
			for t in range(self._tranches[dim][0], self._tranches[dim][1]):
				# Since our filtrations are discrete (i.e. we add exactly one simplex
				# at each time-step), the "degree" of the cell is the same as the time
				# at which it was added.
				cell = t;

				# Create buckets for indices and coefficients; these are created and
				# stored ONCE, but accessed many times.
				faces = Set();
				faceCoefficients = Column();

				# Eliminate.
				faces = self.TwistReducePivotRow(cell, faces, faceCoefficients);
				
				# If we end up with no faces, we've found a pivot row.
				if faces.empty():
					# Mark the face, but only add it to the marked iterable if it's
					# of the right dimension.
					self.marked.insert(cell);
					degree[cell] = t;

					if self._dimensions[cell] == self.homology:
						self.markedIterable[tagged] = cell;
						tagged = tagged + 1;

				# Otherwise, we can just zero out the youngest column.
				else:
					facesIterable = Index();
					facesIterable.insert(facesIterable.begin(), faces.begin(), faces.end());
					self.columnEntries[t] = faces;
					self.columnEntriesIterable[t] = facesIterable;
					self.columnEntriesCoefficients[t] = faceCoefficients;

					youngest = self.youngestOf(faces);
					self.nextColumnAdded[youngest] = t;

					self.columnEntries[youngest] = Set();
					self.columnEntriesIterable[youngest] = Index();
					self.columnEntriesCoefficients[youngest] = Column();

			dim = dim-1;

		for j in range(tagged):
			cell = self.markedIterable[j];

			if self.nextColumnAdded[cell] == 0: events.insert(cell);

		return events


	cpdef SparseLinearCombination TwistBasis(self, INDEXFLAT filtration) noexcept:
		"""
		Reports the linear combinations corresponding to basis elements of the `homology`th
		homology group.

		Args:
			filtration (np.array): Order in which cells are added.

		Returns:
			A dictionary corresponding to linear combinations of basis elements.
		"""
		cdef Set events = self.TwistComputePercolationEvents(filtration);
		cdef SparseLinearCombination basis = SparseLinearCombination();
		
		for e in events:
			basis[e] = self.linearCombinations[e];

		return basis;


	cpdef Set ReduceComputePercolationEvents(self, INDEXFLAT filtration) noexcept:
		"""
		Computes percolation events by pre-computing a basis for the desired
		homology group, then using Gaussian elimination to determine whether any
		of the basis elements are in the image of the map induced by inclusion.

		Args:
			filtration (INDEXFLAT): A NumPy array indicating the order in which
				\(i\)-dimensional cells are to be added.
		
		Returns:
			The set of times at which each element of the basis is found.
		"""
		# If it hasn't already been computed, find the basis for the underlying
		# space.

	
	cpdef Set ComputeGiantCycles(self, INDEXFLAT filtration) noexcept:
		"""
		Computes the times of homological percolation events given a filtration
		and a boundary matrix.

		Args:
			filtration (np.array): Order in which cells are added.

		Returns:
			A `set` of times at which homological percolation occurs.
		"""
		# Flush the set of marked indices, adding premarked ones.
		self.__flushDataStructures();
		cdef Set events = Set();

		# Construct the boundary matrix for this filtration; variables for
		# objects.
		cdef FlatBoundaryMatrix boundary = self.ReorderBoundary(filtration);
		cdef Index facesIterable, degree = Index(self.cellCount);
		cdef Set faces;
		cdef Column faceCoefficients;
		cdef int t, j, cell, dimension, time, tagged, youngest;

		# TODO: shouldn't have to iterate over vertices
		tagged = 0;

		for t in range(0, self.cellCOunt):
			# Since our filtrations are discrete (i.e. we add exactly one simplex
			# at each time-step), the "degree" of the cell is the same as the time
			# at which it was added.
			cell = t;

			# Create buckets for indices and coefficients; these are created and
			# stored ONCE, but accessed many times.
			faces = Set();
			faceCoefficients = Column();

			# Eliminate.
			faces = self.ReducePivotRow(cell, faces, faceCoefficients);
			
			# If we end up with no faces, we've found a pivot row.
			if faces.empty():
				# Mark the face, but only add it to the marked iterable if it's
				# of the right dimension.
				self.marked.insert(cell);
				degree[cell] = t;
				
				if self._dimensions[cell] == self.homology:
					self.markedIterable[tagged] = cell;
					tagged = tagged + 1;

			# Otherwise, store the row in the appropriate locations.
			else:
				youngest = self.youngestOf(faces);
				facesIterable = Index();
				facesIterable.insert(facesIterable.begin(), faces.begin(), faces.end());

				self.columnEntries[youngest] = faces;
				self.columnEntriesIterable[youngest] = facesIterable;
				self.columnEntriesCoefficients[youngest] = faceCoefficients;
				degree[youngest] = t;

		# Once we're done eliminating, check over what we find.
		cdef Set unmarked;

		for j in range(tagged):
			cell = self.markedIterable[j];

			unmarked = self.columnEntries[cell];
			if unmarked.empty(): events.insert(degree[cell]);
		
		return events
	
	
	cpdef Index ComputeBettiNumbers(self, INDEXFLAT subcomplex) noexcept:
		"""
		Computes the _Betti numbers_ — the ranks of the homology groups — of the
		subcomplex specified.

		Args:
			subcomplex (np.ndarray): Indices of cells of the _full_ (flattened)
				boundary matrix to include in the complex.

		Returns:
			A C++ `Index` where the \(i\)th entry is the \(i\)th Betti number.
		"""
		# Construct the boundary matrix for this filtration; variables for
		# objects. Flush the set of marked indices, adding premarked ones.
		cdef FlatBoundaryMatrix subboundary = self.ReindexSubBoundary(subcomplex);
		self.__flushDataStructures(premark=False);

		cdef Index facesIterable, degree = Index(self.cellCount);
		cdef Set faces;
		cdef Column faceCoefficients;
		cdef int t, j, cell, time, tagged, youngest;
		
		tagged = 0;

		for t in range(0, self.cellCount):
			# Since our filtrations are discrete (i.e. we add exactly one simplex
			# at each time-step), the "degree" of the cell is the same as the time
			# at which it was added.
			cell = t;

			# Create buckets for indices and coefficients; these are created and
			# stored ONCE, but accessed many times.
			faces = Set();
			faceCoefficients = Column();

			# Eliminate.
			faces = self.ReducePivotRow(cell, faces, faceCoefficients);
			
			# If we end up with no faces, we've found a pivot row.
			if faces.empty():
				# Mark the face, but only add it to the marked iterable if it's
				# of the right dimension.
				self.marked.insert(cell);
				degree[cell] = t;
				
				self.markedIterable[tagged] = cell;
				tagged = tagged + 1;

			# Otherwise, store the row in the appropriate locations.
			else:
				youngest = self.youngestOf(faces);
				facesIterable = Index();
				facesIterable.insert(facesIterable.begin(), faces.begin(), faces.end());

				self.columnEntries[youngest] = faces;
				self.columnEntriesIterable[youngest] = facesIterable;
				self.columnEntriesCoefficients[youngest] = faceCoefficients;
				degree[youngest] = t;

		# Once we're done eliminating, check over what we find.
		cdef Set unmarked;
		cdef Index bettis = Index(self._tranches.size(),0);

		for j in range(tagged):
			cell = self.markedIterable[j];
			unmarked = self.columnEntries[cell];
			dimension = self._dimensions[cell];

			if unmarked.empty():
				bettis[dimension] = bettis[dimension] + 1
		
		return bettis

