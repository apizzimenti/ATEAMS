
from ateams.complexes import Cubical
from ateams.models import Bernoulli
from ateams import Chain
import os
from pathlib import Path


def construct(L, DIM):
	# Construct complex object.
	fname = Path(f"./data/cubical.{L}.{DIM}.json")
	if not fname.exists():
		fname.parent.mkdir(exist_ok=True, parents=True)
		L = Cubical().fromCorners([L]*DIM)
		L.toFile(fname)
	else:
		L = Cubical().fromFile(fname)

	# Set up Model and Chain.
	SW = Bernoulli(L, dimension=DIM//2)
	N = 1000
	M = Chain(SW, steps=N)

	return M

def chain(M, DESC=""):
	for occupied, giants in M.progress(dynamic_ncols=True, desc=DESC):
		pass

	return M._exitcode

if __name__ == "__main__":
	M = construct(10,4)
	chain(M)

