
# distutils: language=c++

from ..common cimport INDEXFLAT, BoundaryMatrix, Index, Set, Table, Lookup, SparseLinearCombination


cdef class Twist:
	cdef int characteristic, cellCount, dimension, topDimension
	cdef Table addition, multiplication
	cdef Lookup negation, inversion, flatAddition, flatMultiplication

	cdef Index fullBoundary, breaks
	cdef BoundaryMatrix referenceBoundary, workingBoundary

	cdef void __arithmetic(self) noexcept

	cdef BoundaryMatrix FillBoundaryMatrix(self, INDEXFLAT boundary) noexcept
	cdef BoundaryMatrix ReindexBoundaryMatrix(self, INDEXFLAT filtration) noexcept
	cpdef Set ComputePercolationEvents(self, INDEXFLAT filtration) noexcept
	cpdef Set LinearComputePercolationEvents(self, INDEXFLAT filtration) noexcept
	cpdef Set ZpComputePercolationEvents(self, INDEXFLAT filtration) noexcept
	cpdef SparseLinearCombination LinearComputeBasis(self, INDEXFLAT filtration) noexcept
