
from libcpp.vector cimport vector

cdef extern from "fastcomputation.hpp":
	int kernel(vector[int] A, int rank, int nullity, int field)

cpdef pkernel(vector[int] A, int rank, int nullity, int field):
	print(kernel(A, rank, nullity, field))
