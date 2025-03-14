
# cython: language_level=3str, initializedcheck=False, c_api_binop_methods=True, nonecheck=False, profile=True, cdivision=True, wraparound=False, boundscheck=False
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

ctypedef np.int64_t DTYPE_t
DTYPE = np.int64


cdef DTYPE_t Max(DTYPE_t[:] A) noexcept nogil:
	"""
	Substitution for Python's `max` function.

	Args:
		DTYPE_t[:] A: `memoryview` on a Cython NumPy array.

	Returns:
		Maximum value in the array.
	"""
	cdef int j, N;
	cdef DTYPE_t l = A[0];

	N = A.shape[0];

	for j in range(N):
		if A[j] > l: l = A[j]

	return l


cdef DTYPE_t MaxNonzero(DTYPE_t[:,:] A) noexcept nogil:
	"""
	Finds and returns the largest value in the second row of an array given
	the value in the row above it is nonzero.

	Args:
		DTYPE_t[:,:] A: `memoryview` on a Cython NumPy array. It is assumed that
			the values in `A` are "packed" toward the beginning: if there are
			N nonzero values in the second row of the array, they can be found
			in the first N+1 indices.

	Returns:
		Largest value in the second row of an array given the value in the row
		above it is nonzero.
	"""
	cdef int j, N, tripleZeros;
	cdef DTYPE_t l = 0;

	N = A.shape[1];
	tripleZeros = 0;

	for j in range(N):
		if A[0,j] > 0 and A[1,j] > l: l = A[1,j]

		# Avoid traversing the whole array: we check to see whether we've
		# found all the "packed" elements.
		if j > 0:
			if A[1,j-1] == 0 and A[1,j] == 0:
				tripleZeros += 1

		if tripleZeros > 4: break

	return l


cdef DTYPE_t[:] NonzeroIndices(DTYPE_t[:,:] A, DTYPE_t[:] indices) noexcept:
	"""
	Finds and returns an array containing the indices of columns in A whose
	first entry is nonzero.

	Args:
		DTYPE_t[:,:] A: `memoryview` on a Cython NumPy array. It is assumed that
			the values in `A` are "packed" toward the beginning: if there are
			N nonzero values in the second row of the array, they can be found
			in the first N+1 indices.

	Returns:
		Indices of columns with nonzero first entry.
	"""
	cdef int j, N, l, tripleZeros;
	
	N = A.shape[1];
	l = 0;
	tripleZeros = 0;

	for j in range(N):
		if A[0,j] > 0:
			indices[l] = j;
			l += 1

		# Avoid traversing the whole array: we check to see whether we've
		# found all the "packed" elements.
		if j > 0:
			if A[1,j-1] == 0 and A[1,j] == 0:
				tripleZeros += 1

		if tripleZeros > 4: break

	return indices[:l]


cdef DTYPE_t[:,:] SliceMatrix(DTYPE_t[:,:] A, DTYPE_t[:] indices) noexcept:
	"""
	Trims a matrix, keeping only columns with nonzero first entry.

	Args:
		DTYPE_t[:,:] A: `memoryview` on a Cython NumPy array. It is assumed that
			the values in `A` are "packed" toward the beginning: if there are
			N nonzero values in the second row of the array, they can be found
			in the first N+1 indices.
		DTYPE_t[:] indices: `memoryview` on a Cython NumPy array of blanks for
			object recycling.

	Returns:
		`A`, but only with columns with nonzero first entry.
	"""
	cdef int j, t, N;
	cdef DTYPE_t[:] nonzero;
	cdef DTYPE_t[:,:] resized;

	# Find the nonzero indices, create an array of appropriate size, then
	# write. This may be less efficient, but we (sadly) need to do it.
	nonzero = NonzeroIndices(A, indices);
	N = nonzero.shape[0];
	resized = np.empty((2, N), dtype=DTYPE);

	for j in range(N):
		t = nonzero[j]
		resized[0,j] = A[0,t]
		resized[1,j] = A[1,t]

	return resized[:,:N]


cdef int CountNonzero(DTYPE_t[:,:] A) noexcept nogil:
	"""
	Counts the columns of `A` with nonzero first entry.

	Args:
		DTYPE_t[:,:] A: `memoryview` on a Cython NumPy array. It is assumed that
			the values in `A` are "packed" toward the beginning: if there are
			N nonzero values in the second row of the array, they can be found
			in the first N+1 indices.

	Returns:
		Number of columns of `A` with nonzero first entry.
	"""
	cdef int j, N, l, tripleZeros;

	j = 0;
	N = A.shape[1];
	l = 0;
	tripleZeros = 0;

	for j in range(N):
		if A[0,j] > 0: l += 1

		# Avoid traversing the whole array: we check to see whether we've
		# found all the "packed" elements.
		if j > 0:
			if A[1,j-1] == 0 and A[1,j] == 0:
				tripleZeros += 1

		if tripleZeros > 4: break

	return l


cdef void Tare(DTYPE_t[:,:] A) noexcept nogil:
	"""
	Tares (zeroes out) the array.

	Args:
		DTYPE_t[:,:] A: `memoryview` on a Cython NumPy array. It is assumed that
			the values in `A` are "packed" toward the beginning: if there are
			N nonzero values in the second row of the array, they can be found
			in the first N+1 indices.

	Returns:
		Zeroed-out array.
	"""
	cdef int j, N, tripleZeros;

	tripleZeros = 0;
	N = A.shape[1]

	for j in range(N):
		# Avoid traversing the whole array: we check to see whether we've
		# found all the "packed" elements. Do this *first*, though, since we're
		# editing the array to have zero elements precending the current one.
		if j > 0:
			if A[1,j] == 0:
				tripleZeros += 1

		if tripleZeros > 4: break

		A[0,j] = 0
		A[1,j] = 0


cdef DTYPE_t[:,:] FillMarkedIndices (
		DTYPE_t[:,:] A, DTYPE_t[:] powers, list[int] boundary, set[int] marked
	) noexcept:
	"""
	Fills the marked indices of A with the appropriate coefficient.

	Args:
		DTYPE_t[:,:] A: Array of blanks to be filled.
		DTYPE_t[:] powers: Array of alternating powers of -1 (mod p).
		list boundary: Degrees of bounding faces for this plaquette.
		set marked: Marked plaquettes.

	Returns:
		`A` with filled coefficients and indices.
	"""
	cdef int j, b, N;

	N = len(boundary)

	for j in range(N):
		b = boundary[j]
		if b not in marked: continue

		A[0,j] = powers[j]
		A[1,j] = b

	return A


cdef DTYPE_t[:,:] FillSharedIndices(
		DTYPE_t[:,:] coefficients,
		DTYPE_t[:,:] nonpivot,
		set[int] shared,
		dict[int,int] coefficientIndices,
		dict[int,int] nonpivotIndices,
		int q,
		DTYPE_t[:,:] multiplication,
		DTYPE_t[:,:] subtraction
	) noexcept:
	"""
	Computes the coefficients of bounding faces shared by those in `coefficients`
	and `nonpivot`.
	"""
	cdef int k, sub, mul, lindex, rindex;

	for k in shared:
		lindex = coefficientIndices[k]
		rindex = nonpivotIndices[k]
		sub = coefficients[0,lindex]
		mul = multiplication[q,nonpivot[0,rindex]]
		coefficients[0,lindex] = subtraction[sub,mul]

	return coefficients


cdef DTYPE_t[:,:] FillRightIndices(
		DTYPE_t[:,:] coefficients,
		DTYPE_t[:,:] nonpivot,
		set[int] rOnly,
		dict[int,int] nonpivotIndices,
		int q,
		DTYPE_t[:,:] multiplication,
		DTYPE_t[:,:] subtraction
	) noexcept:
	cdef int added, R, a, rindex, mul, N;

	N = coefficients.shape[1];
	R = len(rOnly);
	added = 0;
	a = 0;

	for a in range(N):
		if added == R: break
		if coefficients[0,a] < 1:
			rindex = nonpivotIndices[rOnly.pop()]
			mul = multiplication[q,nonpivot[0,rindex]]
			
			coefficients[0,a] = subtraction[0,mul]
			coefficients[1,a] = nonpivot[1,rindex]

			added += 1

	return coefficients


cdef DTYPE_t[:,:] ReducePivotRow(
		DTYPE_t[:,:] coefficients,
		list[int] boundary,
		set[int] marked,
		DTYPE_t[:] indices,
		dict[int,DTYPE_t[:,:]] dynamicCoeffs,
		dict[int,dict[int,int]] dynamicIndices,
		dict[int,set[int]] dynamicSets,
		DTYPE_t[:] fieldInverses,
		DTYPE_t fieldCharacteristic,
		DTYPE_t[:,:] addition,
		DTYPE_t[:,:] subtraction,
		DTYPE_t[:,:] multiplication,
		DTYPE_t[:] powers
	) noexcept:
	cdef int i, d, dadded, occupied, locator, q;
	cdef dict[int,int] nonpivotIndices, coefficientIndices;
	cdef set[int] left, right, shared, rOnly;
	cdef DTYPE_t[:,:] nonpivot;

	# Fill the coefficient array; count the occupied indices.
	coefficients = FillMarkedIndices(coefficients, powers, boundary, marked)
	occupied = CountNonzero(coefficients)

	while occupied > 0:
		# Find the largest index over the included cells; this is our potential
		# pivot row. Notably, we *cannot* include indices with coefficient zero.
		i = MaxNonzero(coefficients)
		nonpivot = dynamicCoeffs[i]

		# If the column *is* a pivot, then we've already reduced the row and we
		# return the coefficients.
		if nonpivot.shape[1] < 1: break

		# Find the array index of the plaquette of greatest degree (i.e. the
		# plaquette added latest in the filtration).
		nonpivotIndices = dynamicIndices[i]
		locator = nonpivotIndices[i]
		q = fieldInverses[nonpivot[0,locator]]

		# Otherwise, we determine the maximum index over those specified by the
		# chain and get its multiplicative inverse (over the finite field).
		# Because `dynamicIndices` is a Python `dict`, `s` has to remain untyped.
		# `s` is then just a dictionary mapping degrees to indices, so we use this
		# to access the coefficients.
		coefficientIndices = dict()
		left = set()
		dadded = 0;
		
		for d in range(coefficients.shape[1]):
			if dadded == occupied: break
			if coefficients[0,d] > 0:
				coefficientIndices[coefficients[1,d]] = d
				left.add(coefficients[1,d])
				dadded += 1

		# Find the overlap in sets.
		right = dynamicSets[i]
		shared = left & right
		rOnly = right - left
		L = len(left)
		R = len(rOnly)

		# Compute coefficients on shared indices.
		coefficients = FillSharedIndices(
			coefficients, nonpivot, shared, coefficientIndices, nonpivotIndices,
			q, multiplication, subtraction
		)

		# Place the remaining coefficients (and indices) in slots that have been
		# zeroed out.
		coefficients = FillRightIndices(
			coefficients, nonpivot, rOnly, nonpivotIndices, q, multiplication,
			subtraction
		)

		# Count the number of occupied indices and re-start the loop.
		occupied = CountNonzero(coefficients)

	return coefficients


cpdef set[int] computeGiantCyclePairs(
		DTYPE_t[:] times,
		DTYPE_t[:] premarked,
		DTYPE_t[:] dimensions,
		DTYPE_t[:,:] tranches,
		DTYPE_t[:] fieldInverses,
		DTYPE_t fieldCharacteristic,
		int maxDimension,
		int maxIndex,
		DTYPE_t[:,:] empty,
		DTYPE_t[:,:] chain,
		DTYPE_t[:] indices,
		DTYPE_t[:,:] addition,
		DTYPE_t[:,:] subtraction,
		DTYPE_t[:,:] multiplication,
		DTYPE_t[:] powers,
		DTYPE_t[:] filtration,
		list[int] boundary,
		DTYPE_t[:] degree,
	):
	# Buckets for marked indices, dynamic coefficients (for storing 2d arrays
	# corresponding to chains), and indices (mapping cell degrees to indices in
	# `dynamicCoeffs`). The first data structure is required; the second data
	# structure tries to mitigate linear-time searches for max index coefficients.
	cdef set[int] marked, events;
	cdef dict[int,DTYPE_t[:,:]] dynamicCoeffs;
	cdef dict[int,dict[int,int]] dynamicIndices;
	cdef dict[int,set[int]] dynamicSets;

	marked = set(premarked)
	dynamicCoeffs = { }
	dynamicIndices = { }
	dynamicSets = { }

	# Initialize the buckets.
	cdef int _j;

	for _j in range(maxIndex):
		dynamicCoeffs[_j] = empty
		dynamicIndices[_j] = {}
		dynamicSets[_j] = set();

	# Set for collecting the degrees of giant cycles.
	events = set()

	# We want fast loops, so this is how we do it.
	cdef int _t, t, i, cell;
	cdef int N = times.shape[0];
	cdef DTYPE_t[:,:] reduced;

	for _t in range(N):
		t = times[_t]
		cell = filtration[t]
		
		chain = ReducePivotRow(
			chain,
			boundary[cell],
			marked,
			indices,
			dynamicCoeffs,
			dynamicIndices,
			dynamicSets,
			fieldInverses,
			fieldCharacteristic,
			addition,
			subtraction,
			multiplication,
			powers
		)

		reduced = SliceMatrix(chain, indices)

		if reduced.shape[1] < 1:
			marked.add(cell)
			degree[cell] = t
		else:
			i = Max(reduced[1])
			dynamicCoeffs[i] = reduced
			dynamicIndices[i] = dict(zip(reduced[1], range(reduced.shape[1])))
			dynamicSets[i] = set(reduced[1])
			degree[i] = t

		# Zero out the chain.
		Tare(chain)
	
	cdef DTYPE_t[:,:] unmarked;

	for cell in marked:
		if dimensions[cell] != maxDimension-1: continue

		unmarked = dynamicCoeffs[cell]
		if unmarked.shape[1] < 1: events.add(degree[cell])
	
	return events
