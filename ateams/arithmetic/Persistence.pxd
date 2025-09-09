
from ..common cimport FFINT, FLATCONTIG, TABLECONTIG, INDEXTABLE, INDEXFLAT, INDEXTYPE, Index, Set, Column, BoundaryMatrix, FlatBoundaryMatrix, bool, MatrixEntries, SparseLinearCombination

import numpy as np
cimport numpy as np


cdef class Persistence:
	cdef TABLECONTIG addition
	cdef TABLECONTIG subtraction
	cdef TABLECONTIG multiplication
	cdef FLATCONTIG inverse
	cdef FLATCONTIG negation
	cdef FFINT characteristic

	cdef void __arithmetic(self) noexcept
	cdef void __flushDataStructures(self, bool premark=*) noexcept

	# Bounds and 
	cdef INDEXTABLE tranches
	cdef INDEXFLAT dimensions
	cdef INDEXTYPE homology
	cdef INDEXTYPE cellCount
	cdef INDEXTYPE vertexCount
	cdef INDEXTYPE higherCellCount
	cdef INDEXTYPE defaultRowSize
	cdef INDEXTYPE tagged
	cdef INDEXTYPE low
	cdef INDEXTYPE high

	# Parallelization parameters; deprecated.
	cdef INDEXTYPE cores
	cdef INDEXTYPE minBlockSize
	cdef INDEXTYPE maxBlockSize
	cdef bool parallel

	cdef MatrixEntries columnEntries
	cdef FlatBoundaryMatrix columnEntriesIterable
	cdef BoundaryMatrix columnEntriesCoefficients
	cdef BoundaryMatrix linearCombinations

	cdef FlatBoundaryMatrix boundary
	cdef FlatBoundaryMatrix _boundary
	cdef Index _dimensions
	cdef FlatBoundaryMatrix _tranches

	cdef Index markedIterable
	cdef Set marked
	cdef Index premarked
	cdef Index nextColumnAdded
	
	cdef FlatBoundaryMatrix ReorderBoundary(self, INDEXFLAT filtration) noexcept
	cpdef FlatBoundaryMatrix ReindexBoundary(self, INDEXFLAT filtration) noexcept
	cdef FlatBoundaryMatrix ReindexSubBoundary(self, INDEXFLAT subcomplex) noexcept
	cdef FlatBoundaryMatrix Vectorize(self, list[list[int]] flattened) noexcept
	cdef int youngestOf(self, Set column) noexcept

	# Compute persistence via the `twist_reduce` method.
	cdef Set TwistBuildFace(self, int cell, Set &faces, Column &faceCoefficients) noexcept
	cdef Set TwistEliminate(self, int youngest, Set &faces, Column &faceCoefficients, Column &columnsReduced) noexcept
	cdef Set TwistReducePivotRow(self, int cell, Set &faces, Column &faceCoefficients) noexcept
	cpdef Set TwistComputePercolationEvents(self, INDEXFLAT filtration) noexcept
	cpdef SparseLinearCombination TwistBasis(self, INDEXFLAT filtration) noexcept

	# Compute persistence via the standard left-looking method.
	cdef Set RemoveUnmarkedCells(self, int cell, Set &faces, Column &faceCoefficients) noexcept
	cdef Set Eliminate(self, int youngest, Set &faces, Column &faceCoefficients) noexcept
	cdef Set ReducePivotRow(self, int cell, Set &faces, Column &faceCoefficients) noexcept
	cpdef Set ComputePercolationEvents(self, INDEXFLAT filtration) noexcept
	cpdef Set ComputeGiantCycles(self, INDEXFLAT filtration) noexcept
	cpdef Index ComputeBettiNumbers(self, INDEXFLAT subcomplex) noexcept

	# Compute persistence via the basis-reduction method.
	cpdef Set ReduceComputePercolationEvents(self, INDEXFLAT filtration) noexcept
