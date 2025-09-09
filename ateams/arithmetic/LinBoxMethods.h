

#include <ATEAMS/common.h>

// Sample from a kernel.
Index LanczosKernelSample(Index coboundary, int M, int N, int p, int maxTries);

// Varying persistence computers.
Set ComputePercolationEvents(
	Table addition, Table multiplication, Lookup negation, Lookup inversion,
	BoundaryMatrix Boundary, Index breaks, int cellCount
);

Set LinearComputePercolationEvents(
	int field, Lookup addition, Lookup multiplication, Lookup negation,
	Lookup inversion, BoundaryMatrix Boundary, Index breaks, int cellCount,
	int dimension
);

	Set ZpComputePercolationEvents(
	int field, BoundaryMatrix Boundary, Index breaks, int cellCount
);

// Basis computers.
Bases LinearComputeBases(
	int field, Lookup addition, Lookup multiplication, Lookup negation,
	Lookup inversion, BoundaryMatrix Boundary, Index breaks, int cellCount,
	int dimension
);
