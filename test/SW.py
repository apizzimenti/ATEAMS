
import numpy as np
from ateams.complexes import Cubical
from ateams.models import SwendsenWang
from ateams.statistics import constant, critical
from ateams import Chain
import json
import sys
from pathlib import Path


def construct(L, dim, field):
	# Construct complex object.
	fname = Path(f"./data/cubical.{L}.{dim}.json")
	if not fname.exists():
		fname.parent.mkdir(exist_ok=True, parents=True)
		L = Cubical().fromCorners([L]*dim)
		L.toFile(fname)
	else:
		L = Cubical().fromFile(fname)


	# Set up Model and Chain.
	SW = SwendsenWang(L, dimension=dim//2, field=field, temperature=critical(field), _DEBUG=True)
	N = 100
	M = Chain(SW, steps=N)

	return M

def chain(M, DESC=""):
	for (spins, occupied, satisfied) in M.progress(dynamic_ncols=True, desc=DESC):
		# print(spins[M.model.complex.Boundary[2]])
		pass
	return M._exitcode


if __name__ == "__main__":
	M = construct(3, 4, 5)
	chain(M)

