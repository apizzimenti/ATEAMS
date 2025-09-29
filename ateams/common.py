
import numpy as np

MINT = np.int64
"""Global "machine integer" data type; this is equivalent to the C `int` data type."""

FINT = np.int16
"""\"Finite-field integer\" data type; uses 1/4th the space of a machine integer."""


class TooSmallWarning(UserWarning): pass

class NumericalInstabilityWarning(UserWarning): pass


class Matrices:
	boundary = None
	coboundary = None
	full = None


class Bunch(dict):
	def __init__(self, **kwds):
		self.update(kwds)
		self.__dict__ = self

