
# distutils: language=c++

from libcpp.set cimport set as AnySet
from libcpp.vector cimport vector as AnyVector
from libcpp.map cimport map as AnyMap
from libcpp cimport bool
from libc.stdint cimport int32_t

ctypedef int32_t INDEXTYPE
ctypedef char DATATYPE

import numpy as np
cimport numpy as np

# Cython memoryview types. We should be switching to chars (or, really, abandoning
# these types altogether).
ctypedef np.int64_t MINT
ctypedef np.int16_t FFINT
ctypedef FFINT[:] FLAT
ctypedef FFINT[:,:] TABLE
ctypedef MINT[:,:] INDEXTABLE
ctypedef MINT[:] INDEX
ctypedef MINT[::1] INDEXFLAT
ctypedef FFINT[::1] FLATCONTIG
ctypedef FFINT[:,::1] TABLECONTIG

# Base data types.
ctypedef AnySet[INDEXTYPE] Set;
ctypedef AnyMap[INDEXTYPE,INDEXTYPE] Map;
ctypedef AnyVector[INDEXTYPE] Index;

# Lookups and lookup tables.
ctypedef AnyVector[DATATYPE] Lookup;
ctypedef AnyVector[Lookup] Table;

ctypedef AnyMap[INDEXTYPE,DATATYPE] Column;
ctypedef AnyVector[Column] BoundaryMatrix;
ctypedef AnyVector[Index] FlatBoundaryMatrix;
ctypedef AnyVector[Set] MatrixEntries;

# Come up with better names for these.
ctypedef AnyMap[INDEXTYPE,Column] SparseLinearCombination;
ctypedef AnyVector[Index] PersistencePairs;


cdef inline Index Vectorize(INDEXFLAT A) noexcept:
	cdef int i, L;
	cdef Index B;

	L = A.shape[0];
	B = Index(L);

	for i in range(L): B[i] = A[i];

	return B;

