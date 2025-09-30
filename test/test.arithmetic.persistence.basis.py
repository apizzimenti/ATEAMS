
import numpy as np
import galois
# from ateams.common import Bunch

def stringify(A, colbreak=np.inf):
	M, N = A.shape
	S = ""

	for i in range(M):
		for j in range(N):
			if j == colbreak-1: S += "| "
			S += str(A[i,j]) + " "
		S += "\n"

	return S

field = 3
F = galois.GF(field)

Boundaries = {}
Coboundaries = {}

Boundaries[1] = F(np.array([
	[-1,-1,-1,-1,0,0,0,0,0,0,0,0,0],
	[1,0,0,0,-1,-1,-1,0,0,0,0,0,0],
	[0,1,0,0,1,0,0,-1,-1,-1,0,-1,0],
	[0,0,0,0,0,1,0,1,0,0,-1,0,0],
	[0,0,1,0,0,0,0,0,1,0,0,0,0],
	[0,0,0,1,0,0,1,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,1,1,0,0],
	[0,0,0,0,0,0,0,0,0,0,0,0,-1],
	[0,0,0,0,0,0,0,0,0,0,0,1,1]
])%field)

Coboundaries[0] = Boundaries[1][::].T

Boundaries[2] = F(np.array([
	[1,0,0,0],
	[0,1,0,0],
	[0,-1,0,0],
	[-1,0,0,0],
	[0,0,1,0],
	[0,0,-1,0],
	[1,0,0,0],
	[0,0,1,0],
	[0,1,0,0],
	[0,0,0,0],
	[0,0,0,1],
	[0,0,0,-1],
	[0,0,0,1]
])%field)

Coboundaries[1] = Boundaries[2][::].T

# A basis for the cycle space is...
CycleBasis = F(np.array([
	[1,-1,0,0,1,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,1,0,1,-1,0,0]
])%field)

# A basis for the boundary space is...
BoundaryBasis = Boundaries[2].T

# Now stack the cycle and boundary bases on top of one another and solve?
stacked = F(np.r_[CycleBasis,BoundaryBasis])

CocycleBasis = []

for t in range(len(CycleBasis)):
	_f = F.Zeros(stacked.shape[0])
	_f[t] = 1

	augmented = F(np.c_[stacked, _f])
	reduced = augmented.row_reduce()
	coeffs = reduced[:,-1]
	f = F.Zeros(stacked.shape[1])

	# Find the pivots; zero out the appropriate values and set the others.
	pivots = (reduced!=0).argmax(axis=1)
	for q in range(len(coeffs)): f[pivots[q]] = coeffs[q]

	CocycleBasis.append(F(f))

CocycleBasis = F(CocycleBasis)

# Now actually try to persist!
S = ""

for t in range(0,len(Coboundaries[0])):
	partial = F(np.c_[Coboundaries[0], CocycleBasis.T])[:t+1]
	coboundary = Coboundaries[0][:t+1]
	cobasis = CocycleBasis.T[:t+1]

	partialrank = len(partial.column_space())
	coboundaryrank = len(coboundary.column_space())
	cobasisrank = len(cobasis.column_space())

	S += f"step {t+1}, adjoined rank {partialrank} \n"
	S += stringify(partial.row_reduce(), partial.shape[1]-1)
	S += "\n\n"

	# S += f"step {t+1}, coboundary {coboundaryrank} \n"
	# S += stringify(coboundary.row_reduce())
	# S += "\n\n"

	# S += f"step {t+1}, cobasis rank {cobasisrank} \n"
	# S += stringify(cobasis.row_reduce())
	# S += "\n\n"

	diff = partialrank-coboundaryrank
	S += f"rank diff: {partialrank}-{coboundaryrank} = {diff}\n\n"

	S += "#########################\n\n"

with open("data/adjoined.txt", "w") as w: w.write(S) 
