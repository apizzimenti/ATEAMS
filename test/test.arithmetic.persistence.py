
import numpy as np
from time import time

from ateams.complexes import Cubical
from ateams.arithmetic import Twist


def stringify(A, colbreak):
	M, N = A.shape
	S = ""

	for i in range(M):
		for j in range(N):
			if j == colbreak-1: S += "| "
			S += str(A[i,j]) + " "
		S += "\n"

	return S


field = 3
dimension = 2
Cubes = Cubical().fromCorners([4,4,4,4])
boundary, coboundary = Cubes.recomputeBoundaryMatrices(dimension)
print(Cubes.breaks)

P = Twist(field, Cubes.matrices.full, Cubes.breaks, len(Cubes.flattened), dimension)
filtration = np.arange(len(Cubes.flattened))

lstart = time()
ltimes = P.LinearComputePercolationEvents(filtration)
lstop = time()

right = [t for t in ltimes if Cubes.breaks[dimension] < t < Cubes.breaks[dimension+1]]
print(lstop-lstart)
print(right)
print()

# bstart = time()
# btimes = P.RankComputePercolationEvents(filtration)
# bstop = time()
# print(bstop-bstart)
# print(btimes)

