
from setuptools import setup, Extension
from distutils import sysconfig
from Cython.Build import cythonize
import os
import subprocess
import numpy

#########################
### INSTALLATION NOTE ###
#########################

# PHAT is often *very difficult* to install. Before installing, please ensure
# that the following are installed and accessible in your system:
#
#   • PyBind11, setuptools, wheel (via pip)
#   • g++ (preferably >12)
#
# When installing PHAT, please use the arguments
#
#   pip install --use-deprecated=legacy-resolver --no-binary :all: phat
#
# To ensure installation.

os.environ["CC"] = "clang++"
os.environ["CXX"] = "clang++"
os.environ['LDSHARED'] = 'clang -shared'

extensions = [
	Extension(
		"*",
		["ateams/arithmetic/matrices.pyx", "ateams/arithmetic/fastcomputation.cpp"],
		include_dirs=[
			"/usr/local/include/givaro/include",
			"/usr/local/include/gmp/include",
			"/usr/local/include/linbox/include",
			"/usr/local/include/fflas-ffpack/include"
		],
		library_dirs=[
			"/usr/local/include/givaro/lib",
			"/usr/local/include/gmp/lib",
			"/usr/local/include/linbox/lib",
			"/usr/local/include/fflas-ffpack/lib"	
		],
		extra_compile_args=["-std=c++11"],
		language="c++"
	),
	Extension(
		"*",
		[
			"ateams/arithmetic/linearAlgebra.pyx",
			"ateams/arithmetic/persistence.pyx",
			"ateams/arithmetic/cubicalComplex.pyx"
		],
		include_dirs=[numpy.get_include()],
		language="c"
	)
]

setup(
    ext_modules=cythonize(
		extensions,
		annotate=True,
		language_level="3"
	)
)