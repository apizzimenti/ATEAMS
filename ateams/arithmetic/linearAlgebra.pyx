
# cython: language_level=3str, initializedcheck=False, c_api_binop_methods=True, nonecheck=False, profile=True, cdivision=True, wraparound=False, boundscheck=False
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.ccall
def isNullHomologous(A, x, I, includes=[]):
	"""
	Checks whether the provided sequence of faces is a boundary of some chain of
	plaquettes.

	Args:
		A (galois.Array): Complete boundary matrix.
		x (galois.Array): Chain of faces.
		I (galois.Array): Identity matrix of dimension \((n+1) \\times (n+1)\),
			where \(n\) is the number of columns of \(A\).
		includes (iterable=[]): Column indices to _include_.

	Returns:
		A boolean which is truthy when \(x\) is nullhomologous and falsy otherwise.
	"""
	if len(includes): B = A.take(includes, axis=1)
	else: B = A
	
	# Augment the matrix, get the RREF, and check whether we have a solution or
	# not. If not, the RREF of the matrix is the identity of dimension one
	# larger than the boundary submatrix.
	R = (np.c_[B, x].astype(np.int32)).row_reduce()
	_, m = B.shape
	return not (R[:m+1,:] == I[:m+1,:m+1]).all()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.ccall
def sampleFromKernel(A, F, includes=[], relativeCells=[], relativeFaces=[]):
	"""
	Uniformly randomly samples a cochain given the coboundary matrix A.

	Args:
		A (np.ndarray): Complete coboundary matrix.
		F (galois.GF): Finite field representation.
		includes (iterable=[]): Column indices to _include_.
		relativeCells (iterable=None): Column indices to _include_. (Possible
			duplicate of the above?)
		relativeFaces (iterable=None): Row indices to _include_.

	Returns:
		A cochain on the \((k-1)\)-skeleton of the complex that is a cocycle on
		whatever subcomplex is specified by `includes`, `relativeCells`, and
		`relativeFaces`, represented as a `galois.FieldArray`.
	"""
	# If we're including all the columns, just set our submatrix equal to the
	# whole coboundary; otherwise, exclude some of the columns.
	if len(includes): B = A.take(includes, axis=0)
	else: B = A

	# Relative homology.
	if len(relativeCells): C = B.take(relativeCells, axis=0)
	else: C = B

	if len(relativeFaces): D = C.take(relativeFaces, axis=1)
	else: D = C
	
	# Find a basis for the kernel of B, then take a uniformly random linear
	# combination of the basis elements; this is equivalent to uniformly
	# randomly sampling a cocycle.
	K = D.null_space()
	M, _ = K.shape
	Q = F.Random(M)
	return (Q@K)


@cython.boundscheck(False)
@cython.wraparound(False)
def evaluateCochain(np.ndarray[np.int_t, ndim=2] boundary, spins):
	"""
	Evaluates the coboundary of a cochain on the \(k\)-skeleton.

	Args:
		boundary (np.ndarray): Boundary of the \(k\)-skeleton.
		spins (galois.FieldArray): Cochain (spins) on the \((k-1)\)-skeleton.
	
	Returns:
		A `galois.FieldArray` of coefficients on the \(k\)-skeleton.
	"""
	evaluation = spins[boundary]
	evaluation[:, 1::2] = -evaluation[:, 1::2]
	return np.array(evaluation.sum(axis=1), dtype=int)


@cython.boundscheck(False)
@cython.wraparound(False)
def autocorrelation(np.ndarray[np.float_t, ndim=1] data):
	"""
	Computes the autocorrelation of a given observable.

	Args:
		data (Iterable): An iterable, indexed by sample times, containing data
			from a given observable.
	
	Returns:
		An `np.ndarray` of autocorrelation data.
	"""

	# Expected value (i.e. sample mean, which converges to the expectation by
	# LLN).
	cdef float mu = data.mean();
	cdef float normalized = data-mu;
	cdef float N = len(data);

	autocorrs = np.array([
		np.dot(normalized[t:], normalized[:N-t])*(1/N) for t in range(N)
	])

	return autocorrs/autocorrs[0]
