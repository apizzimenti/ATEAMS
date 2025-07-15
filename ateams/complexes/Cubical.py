
import json
import numpy as np
from ast import literal_eval as le
from pathlib import Path
from itertools import combinations as combs, product
from math import comb

from ..common import MINT, Matrices
from .Cubical import boundaryMatrices, fullBoundaryMatrix


class Matrices:
	boundary = None
	coboundary = None


class Cubical:
	def __init__(self): pass

	def fromCorners(self, corners, periodic=True):
		"""
		Creates a cell complex with the given corners made of cells of the
		provided dimension.

		Args:
			corners (list): Corners of the complex; determines the maximal
				cell dimension.
			periodic (bool): Do we use periodic boundary
				conditions (i.e. are we making a torus)?
		"""
		self._construct(corners, periodic)
		return self
	

	@staticmethod
	def _fullBoundaryMatrix(flattened, coefficients):
		return np.array(fullBoundaryMatrix(flattened, coefficients))
	

	def _construct(
			self, corners, periodic, data=None
		):
		self.dimension = len(corners)
		self.periodic = periodic
		self.corners = corners
		self.matrices = Matrices()

		# Specify sparse "boundary matrices," which we'll convert to *actual*
		# matrices for the purposes of our stuff.
		if not data:
			vertexMap, Complex, flatBoundary, flatCoboundary = cubicalComplex(self.corners, self.dimension, self.periodic)
			self.vertexMap, self.Boundary = vertexMap, Complex
		else:
			self.vertexMap = data.get("vertexMap", None)
			self.Boundary = {
				int(t): data["Complex"][t].astype(int) for t in data["Complex"].files
			}
			data["Complex"].close()
			flatBoundary, flatCoboundary = boundaryMatrix(self.Boundary, self.dimension)
		
		# Construct the finite field and boundary matrices.
		self.reindexer, self.reindexed, self.flattened = flatten(self.Boundary, self.dimension)
		self.matrices.boundary = flatBoundary
		self.matrices.coboundary = flatCoboundary

		self.orientations = coefficients = { d: np.tile([-1,1], d*2).astype(MINT) for d in range(1, self.dimension+1) }
		self.matrices.full = np.array(fullBoundaryMatrix(self.flattened, coefficients), dtype=MINT)

		# Get index ranges. (This may turn out to be vestigial; who knows?)
		self.tranches = np.zeros((self.dimension+1, 2), dtype=int)
		self.tranches[0][1] = len(self.Boundary[0])

		for d in range(1, self.dimension+1):
			self.tranches[d] = [self.tranches[d-1][1], self.tranches[d-1][1] + len(self.Boundary[d])]

		# Breaks. (For percolation computation.)
		self.breaks = np.array(self.tranches[:,0])

		return self
	

	def recomputeBoundaryMatrices(self, dimension):
		"""
		Recomputes the boundary matrices up to the provided dimension.

		Args:
			dimension (int): New dimention.

		Returns:
			A triple of matrices --- the new flat boundary/coboundary, and the
			new full boundary.
		"""
		boundary, coboundary = boundaryMatrix(self.Boundary, dimension)
		
		return boundary, coboundary


	
	def toFile(self, fp, vertexMap=False):
		"""
		JSON-serializes this object and writes it to file so we can reconstruct
		it later.

		Args:
			fp (str): Filepath.
			vertexMap (bool): Do we save the vertex map? Not doing so saves
				a lot of space.
		"""
		# Write compressed boundary matrix and vertex maps to file.
		absolute = Path(fp).resolve()
		root = absolute.parent
		stem = absolute.stem
		ComplexFile = root/f".{stem}.complex.npz"
		np.savez(ComplexFile, **{str(t): v for t, v in self.Boundary.items()})

		with open(fp, "w") as write:
			json.dump(
				{
					"dimension": self.dimension,
					"periodic": int(self.periodic),
					"corners": self.corners,
					"Complex": str(ComplexFile),
					"vertexMap": { str(k): int(v) for k, v in self.vertexMap.items() } if vertexMap else { }
				}, write
			)

	
	def fromFile(self, fp:str, vertexMap=False):
		"""
		Reconstructs a serialized Complex.

		Args:
			fp (str): Filepath.
			vertexMap (bool): Is there a vertexmap to load?
		"""
		with open(fp, "r") as read:
			# Read file into memory.
			serialized = json.load(read)

			# Set field and dimension.
			periodic = bool(serialized["periodic"])
			corners = serialized["corners"]
			data = { "Complex": np.load(serialized["Complex"], allow_pickle=True) }
			if vertexMap: data["vertexMap"] = { le(k): int(v) for k, v in serialized["vertexMap"].items() }

			# Construct indices, boundary matrix, and graph.
			return self._construct(
				corners, periodic, data
			)



def flatten(Complex, D):
	"""
	Flattens the given Complex for a PHAT-style specification.

	Args:
		Complex (dict): Dictionary sending dimensions (keys) to boundary matrices
			(values).
		D (int): Top-level dimension of the complex.

	Returns:
		A triplet: the first item is a reindexer dictionary, taking each dimension
		to a count of all faces of lower dimension; the second is the re-indexed
		`Complex` object; the third is a flattened, reindexed `Complex` object
		(for use with PHAT).
	"""
	# Line up indices.
	reindexer = {
		d: np.arange(len(Complex[d])) + sum([len(Complex[t]) for t in range(d)])
		if d > 0 else np.arange(len(Complex[0]))
		for d in range(D+1)
	}

	# Reindex.
	reindexed = {
		d: reindexer[d-1][Complex[d]]
		for d in range(1, D+1)
	}

	reindexed[0] = Complex[0]

	# Flatten.
	flattened = sum([reindexed[t].tolist() for t in range(D+1)], [])

	return reindexer, reindexed, flattened


def boundaryMatrix(Complex, D):
	"""
	Constructs the boundary and coboundary matrices in flat format.

	Args:
		Complex (dict): Dictionary of numpy arrays corresponding to the boundaries
			of each cell.
		D (int): Dimension of the complex.
	"""
	# Alternating coefficients for bounadry components.
	coefficients = np.ones(len(Complex[D]))
	coefficients[::2] = -1
	coefficients = coefficients.astype(MINT)

	# Compute matrices.
	Complex[D] = Complex[D].astype(MINT)
	matrices = boundaryMatrices(Complex[D], coefficients)
	boundary = matrices[0]
	coboundary = matrices[1]

	return np.array(boundary, dtype=MINT), np.array(coboundary, dtype=MINT)


def hammingCube(D):
	"""
	Constructs a Hamming cube of dimension D.

	Args:
		D (int): Overall dimension.
		cutoff (int): Top dimension.

	Returns:
		A dictionary of NumPy arrays corresponding to (compressed) boundary
		matrices at each dimension.
	"""
	# Bucket for the boundary.
	Complex = {}

	# Specify the vertices and edges, building everything else from there.
	basis = np.array([2**k for k in range(0, D)])
	vertices = np.array(range(0, 2**D))
	Complex[0] = vertices.astype(MINT)
	limit = max(vertices)

	edges = np.array([
		np.array([v, v+b])
		for v, b in product(vertices, basis)
		if v != b and v+b <= limit and v&(v+b) == v
	])
	Complex[1] = edges.astype(MINT)

	for d in range(2, D+1):
		# Construct an appropriately-sized bucket for the boundary of the cube.
		cubes = 2**(D-d)*comb(D, d)
		facesPerCube = 2*d
		_boundary = np.zeros((cubes, facesPerCube), dtype=MINT)

		# Get all the combinations of things in the previous skeleton and
		# filter them; we effectively check that the composition-of-boundary-maps
		# condition holds by requiring every element of the (d-2) skeleton to
		# appear *exactly twice* in whatever we're constructing.
		combinations = np.array(list(combs(range(len(Complex[d-1])), facesPerCube)))\
			.flatten()\
			.reshape((-1, facesPerCube))
		
		filled = 0
		
		for combination in combinations:
			faces = Complex[d-1][combination]
			indices = faces.flatten()
			
			if len(np.unique(indices)) == len(indices)/2:
				_boundary[filled] = combination
				filled += 1

		Complex[d] = _boundary.astype(MINT)

	return Complex



def cubicalComplex(corners, D, periodic=True):
	"""
	Constructs a cubical complex completely specified by a boundary matrix and
	a vertex map.

	Args:
		corners (iterable): An iterable specifying the extent of the complex. If
			`periodic` is truthy, antipodal vertices are identified; for example,
			if `corners = [3,3]`, then the vertices at indices (0,0), (0,3), (3,0),
			and (3,3) are identified.
		D (int): Top-level dimension.
		periodic (bool=True): If truthy, we construct a \(d\)-fold torus, where \(d\)
			is the topmost dimension of the complex (as determined by the number
			of corners specified). For example, if `periodic` is falsy, then
			`corners = [3,3]` hands back a 4x4 grid (zero-indexed).

	Returns:
		A quadruplet containing
		
		1. a `dict` mapping vertex indices to integer coordinates

		2. a `dict` mapping dimensions to boundary matrices

		3. a numpy array representing the boundary matrix (in flattened form)
			for the specified dimension;

		4. the same thing as (3), except it's the coboundary matrix (so the 0th
			and 1st terms mod 3 are swapped).
	"""
	# Get the boundary specification for a D-dimensional Hamming cube; we use
	# this cube to enforce an ordering on the vertices (and thus edges, etc.) so
	# the boundary/coboundary operations make sense.
	boundary = hammingCube(D)

	# Specify the "basis" Hamming cube and the remaining separately; this
	# helps when we need to shift the cubes around and properly index.
	_basis = list(product(*[range(2) for c in corners]))
	basis = np.array(_basis)

	i = 1
	_extra = list(product(*[range(c+i) for c in corners]))
	extra = np.array([v for v in _extra if v not in _basis])

	# Construct a complete list of coordinates and indices; create a *dict* to
	# make creation a bit easier.
	vertices = np.concatenate([basis, extra])
	indices = np.arange(len(vertices))
	tups = [tuple(v) for v in vertices]

	verticesToIndices = { tups[t]: t for t in indices }

	if periodic:
		# Identify antipodal vertices.
		for v in vertices: verticesToIndices[tuple(v)] = verticesToIndices[tuple(v%corners)]

		# After identification, count the unique values we see, then re-map.
		unique = np.unique(list(verticesToIndices.values()))
		vertices = np.arange(len(unique))
		indices[unique] = vertices
		verticesToIndices = { t: indices[verticesToIndices[t]] for t in tups }
		
	# Explode and do some reshaping. We need everything in terms of their
	# local coordinates, so we'll have to do some re-indexing on the higher-
	# dimensional faces.        
	Complex = {}
	Complex[0] = vertices
	Skeleta = {}

	anchors = np.array(list(product(*[range(c) for c in corners])))
	anchorMapping = { tuple(a): { d: [] for d in range(D+1) } for a in anchors }

	# Handle edges first, then reindex.
	for anchor in anchors:
		stacked = np.array([anchor]*len(basis))
		localIndices = stacked+basis
		localFaces = np.array([verticesToIndices[tuple(v)] for v in localIndices])

		anchorMapping[tuple(anchor)][0] = localFaces
		anchorMapping[tuple(anchor)][1] = localFaces[boundary[1]]

	allEdges = np.concatenate([a[1] for a in anchorMapping.values()])
	uniqueEdges = np.unique(allEdges, axis=0)
	Complex[1] = uniqueEdges
	Skeleta[1] = { tuple(e): j for j, e in enumerate(uniqueEdges) }

	# Now do everything else! We shouldn't get overlaps here.
	for d in range(2, D+1):
		for anchor in anchors:
			localFaces = anchorMapping[tuple(anchor)][d-1]
			localBoundary = np.zeros(boundary[d].shape).astype(int)

			for j in range(len(boundary[d])):
				localized = np.array([Skeleta[d-1][tuple(localFaces[f])] for f in boundary[d][j]])
				localBoundary[j] = localized

			anchorMapping[tuple(anchor)][d] = localBoundary
		
		allBoundaries = np.concatenate([a[d] for a in anchorMapping.values()])
		uniqueBoundaries = np.unique(allBoundaries, axis=0)
		Complex[d] = uniqueBoundaries
		Skeleta[d] = { tuple(f): j for j, f in enumerate(uniqueBoundaries) }

		# Delete repeated keys in the vertex mapping.
		vti = {}
		seen = set()

		for v, k in verticesToIndices.items():
			if k not in seen: vti[v] = k
			seen.add(k)

		verticesToIndices = vti

	return verticesToIndices, Complex, *boundaryMatrix(Complex, D)

