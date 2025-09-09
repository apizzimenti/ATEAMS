
from ..common cimport Table, Lookup, BoundaryMatrix, Index, Set, Bases

cdef extern from "LinBoxMethods.h":
	# Sample computers.
	Index LanczosKernelSample(
		Index coboundary, int M, int N, int p, int maxTries
	) except +

	# Persistence computers.
	Set ComputePercolationEvents(
		Table addition, Table multiplication, Lookup negation, Lookup inversion,
		BoundaryMatrix Boundary, Index breaks, int cellCount
	) noexcept

	Set LinearComputePercolationEvents(
		int field, Lookup addition, Lookup multiplication, Lookup negation,
		Lookup inversion, BoundaryMatrix Boundary, Index breaks, int cellCount,
		int dimension
	) noexcept

	Set ZpComputePercolationEvents(
		int field, BoundaryMatrix Boundary, Index breaks, int cellCount
	) noexcept

	# Basis computers.
	Bases LinearComputeBases(
		int field, Lookup addition, Lookup multiplication, Lookup negation,
		Lookup inversion, BoundaryMatrix Boundary, Index breaks, int cellCount,
		int dimension
	) noexcept

