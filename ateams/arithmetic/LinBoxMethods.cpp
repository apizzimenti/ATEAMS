
#include <linbox/linbox-config.h>
#include <linbox/solutions/solve.h>
#include <linbox/matrix/sparse-matrix.h>
#include <linbox/ring/modular.h>
#include <linbox/field/gf2.h>
#include <linbox/blackbox/zo-gf2.h>
#include <iostream>

#include "LinBoxMethods.h"

using namespace std;

typedef Givaro::Modular<int> Zp;
typedef LinBox::SparseMatrix<Zp, LinBox::SparseMatrixFormat::SparseSeq> ZpMatrix;
typedef LinBox::DenseVector<Zp> ZpVector;
typedef LinBox::ZeroOne<LinBox::GF2> Z2Matrix;
typedef LinBox::DenseVector<LinBox::GF2> Z2Vector;


template <typename Vector>
bool containsNonzero(Vector X) {
	// Check whether there are nonzero elements in the VVector.

	for (auto it = X.begin(); it != X.end(); ++it) {
		if (*it > 0) { return true; }
	}

	return false;
}

template <typename Matrix, typename Field>
Matrix FieldFill(Index coboundary, int M, int N, Field F) {
	// Construct the sparse coboundary matrix.
	Matrix A(F, M, N);

	for (int t = 0; t < coboundary.size(); t += 3) {
		typename Field::Element q;
		int i, j;

		i = coboundary[t];
		j = coboundary[t+1];
		F.init(q, coboundary[t+2]);
		A.setEntry(i,j,q);
	}

	return A;
}

template <typename Matrix, typename Vector>
Index populate(Matrix A, Vector X) {
	// Populate a vector with entries from a LinBox vector.
	Index x(A.coldim());
	
	for (size_t k = 0; k < A.coldim(); k++) {
		x[k] = X.getEntry(k);
	}

	return x;
}


Index LanczosKernelSample(Index coboundary, int M, int N, int p, int maxTries) {
	// Construct the finite field and construct the matrix. If we're over Z/2Z,
	// use specialized matrices for our operations.
	typedef ZpMatrix Matrix;
	typedef ZpVector Vector;
	typedef Zp Field;

	// if (p < 3) {
	// 	typedef Z2Matrix Matrix;
	// 	typedef Z2Vector Vector;
	// 	typedef LinBox::GF2 Field;
	// }

	Field F(p);
	Matrix A = FieldFill<Matrix,Field>(coboundary, M, N, F);
	ZpVector X(F, A.coldim()), b(F, A.rowdim());
	
	// Preconditioners in order of strength. We try all but FullDiagonal twice;
	// if a zero result still occurs, we sample with the FullDiagonal for the
	// remaining attempts.
	LinBox::Method::Lanczos LANC;
	
	vector<LinBox::Preconditioner> Preconditioners({
		LinBox::Preconditioner::None,
		LinBox::Preconditioner::PartialDiagonal,
		LinBox::Preconditioner::PartialDiagonalSymmetrize,
		LinBox::Preconditioner::FullDiagonal
	});

	int t = 0, pc = 0, tried = 0, pcs = Preconditioners.size();

	while (!containsNonzero(X) && t < maxTries) {
		if (pc < pcs-1) {
			tried = 0;

			while (tried < 2) {
				LANC.preconditioner = Preconditioners[pc];
				solve(X, A, b, LANC);
				t++;
				tried++;
			}
			pc++;
		} else {
			LANC.preconditioner = LinBox::Preconditioner::FullDiagonal;
			solve(X, A, b, LANC);
			t++;
		}
	}

	return populate(A, X);
}


void Reduce(Index boundary, int M, int N, int p) {
	// Construct the finite field and construct the matrix. If we're over Z/2Z,
	// use specialized matrices for our operations.
	typedef ZpMatrix Matrix;
	typedef ZpVector Vector;
	typedef Zp Field;

	Field F(p);
	Matrix A = FieldFill<Matrix,Field>(boundary, M, N, F);
	ZpVector X(F, A.coldim()), b(F, A.rowdim());

	// Check whether there's a nonzero solution.
	cout << "here" << endl;
}


template <typename BalancedStorage>
int youngestOf(BalancedStorage column) {
	// Gets the "youngest" (largest-indexed) cell in the column.
	return column.rbegin()->first;
}


typedef map<int,Zp::Element> ZpColumn;
typedef vector<ZpColumn> ZpBoundaryMatrix;


ZpBoundaryMatrix ZpFillBoundaryMatrix(BoundaryMatrix Boundary, Zp Field) {
	ZpBoundaryMatrix B(Boundary.size());
	Zp::Element q;
	Column column;
	ZpColumn pcolumn;
	int face;

	for (int t = 0; t < Boundary.size(); t++) {
		column = Boundary[t];
		pcolumn = ZpColumn();

		for (auto it=column.begin(); it != column.end(); it++) {
			face = it->first;
			Field.init(q, it->second);
			pcolumn[face] = q;
		}

		B[t] = pcolumn;
	}

	return B;
}


Set ZpComputePercolationEvents(int field, BoundaryMatrix _boundary, Index breaks, int cellCount) {
	Zp F(field);
	ZpBoundaryMatrix Boundary = ZpFillBoundaryMatrix(_boundary, F);

	Index nextColumnAdded = Index(cellCount, 0);
	ZpColumn cell, youngest;
	Set marked = Set();
	int face, high, numBreaks = breaks.size();

	Zp::Element q, r, s, inv, neg, prod, result;
	F.init(q);
	F.init(r);
	F.init(s);
	// char q, r, s, inv, prod, result;

	for (int d = numBreaks-1; d > 0; d--) {

		high = (d+1 >= numBreaks ? cellCount : breaks[d+1]);

		for (int j = breaks[d]; j < high; j++) {
			// If we're of the wrong dimension, keep going.
			// if (dim(j, breaks) != d) { continue; }
			cell = Boundary[j];

			while (!cell.empty() && nextColumnAdded[youngestOf(cell)] != 0) {
				// Get the "youngest" cell in the boundary and subtract it from
				// the current cell.
				youngest = Boundary[nextColumnAdded[youngestOf(cell)]];

				// Get the multiplicative inverse of the coefficient and do
				// arithmetic over the row.
				q = youngest[youngestOf(cell)];
				F.inv(inv, q);

				for (auto it=youngest.begin(); it != youngest.end(); ++it) {
					face = it->first;
					s = it->second;

					// Take the product of inv(q) with the entry of this row;
					// if this is row youngestOf(cell), then this product is 1
					// (it's a pivot). If the current cell shares this face, do
					// the subtraction; otherwise, just add a new coefficient.
					F.mul(prod, inv, s);
					// prod = multiplication[inv][s];

					if (cell.count(face) > 0) {
						F.sub(result, cell[face], prod);
						// result = addition[cell[face]][negation[prod]];

						if (F.isZero(result)) {
							cell.erase(face);
						} else {
							cell[face] = result;
						}
					} else {
						F.neg(result, prod);
						cell[face] = result;
					}
				}
			}
			// Check whether we've eliminated the column. For some god damn reason
			// we have to re-set the entry of the Boundary??? Why?????? Scope??? wtf
			if (!cell.empty()) {
				Boundary[j] = cell;
				nextColumnAdded[youngestOf(cell)] = j;
				Boundary[youngestOf(cell)] = ZpColumn();
			} else {
				marked.insert(j);
			}
		}
	}

	// Find the essential birth times by checking whether the column is marked
	// (i.e. is a cycle) and has no younger columns to add. (For some reason,
	// 0 gets left out here. Not sure why...)
	Set essential = Set();
	essential.insert(0);

	for (auto it = marked.begin(); it != marked.end(); it++) {
		if (nextColumnAdded[*it] == 0) essential.insert(*it);
	}

	return essential;
}





Set ComputePercolationEvents(
		Table addition, Table multiplication, Lookup negation, Lookup inversion,
		BoundaryMatrix Boundary, Index breaks, int cellCount
	) {
	Index nextColumnAdded = Index(cellCount, 0);
	Column cell, youngest;
	Set marked = Set();
	int face, high, numBreaks = breaks.size();

	char q, r, s, inv, prod, result;

	for (int d = numBreaks-1; d > 0; d--) {

		high = (d+1 >= numBreaks ? cellCount : breaks[d+1]);

		for (int j = breaks[d]; j < high; j++) {
			// If we're of the wrong dimension, keep going.
			// if (dim(j, breaks) != d) { continue; }
			cell = Boundary[j];

			while (!cell.empty() && nextColumnAdded[youngestOf(cell)] != 0) {
				// Get the "youngest" cell in the boundary and subtract it from
				// the current cell.
				youngest = Boundary[nextColumnAdded[youngestOf(cell)]];

				// Get the multiplicative inverse of the coefficient and do
				// arithmetic over the row.
				q = youngest[youngestOf(cell)];
				inv = inversion[q];

				for (auto it=youngest.begin(); it != youngest.end(); ++it) {
					face = it->first;
					s = it->second;

					// Take the product of inv(q) with the entry of this row;
					// if this is row youngestOf(cell), then this product is 1
					// (it's a pivot). If the current cell shares this face, do
					// the subtraction; otherwise, just add a new coefficient.
					prod = multiplication[inv][s];

					if (cell.count(face) > 0) {
						result = addition[cell[face]][negation[prod]];

						if (result < 1) {
							cell.erase(face);
						} else {
							cell[face] = result;
						}
					} else {
						cell[face] = negation[prod];
					}
				}
			}
			// Check whether we've eliminated the column. For some god damn reason
			// we have to re-set the entry of the Boundary??? Why?????? Scope??? wtf
			if (!cell.empty()) {
				Boundary[j] = cell;
				nextColumnAdded[youngestOf(cell)] = j;
				Boundary[youngestOf(cell)] = Column();
			} else {
				marked.insert(j);
			}
		}
	}

	// Find the essential birth times by checking whether the column is marked
	// (i.e. is a cycle) and has no younger columns to add. (For some reason,
	// 0 gets left out here. Not sure why...)
	Set essential = Set();
	essential.insert(0);

	for (auto it = marked.begin(); it != marked.end(); it++) {
		if (nextColumnAdded[*it] == 0) essential.insert(*it);
	}

	return essential;
}

typedef function<DATATYPE(DATATYPE, DATATYPE)> binop;
typedef function<DATATYPE(DATATYPE)> unop;


binop _add(Lookup addition, DATATYPE p) {
	return [addition, p](DATATYPE a, DATATYPE b) { return addition[a*p + b]; };
}

binop _mult(Lookup multiplication, DATATYPE p) {
	return [multiplication, p](DATATYPE a, DATATYPE b) { return multiplication[a*p + b]; };
}

unop _neg(Lookup negation) {
	return [negation](DATATYPE a) { return negation[a]; };
}

unop _inv(Lookup inversion) {
	return [inversion](DATATYPE a) { return inversion[a]; };
}


Set LinearComputePercolationEvents(
		int field, Lookup addition, Lookup multiplication, Lookup negation, Lookup inversion,
		BoundaryMatrix Boundary, Index breaks, int cellCount, int dimension
	) {
	Index nextColumnAdded = Index(cellCount, 0);
	Column cell, youngest;
	Set marked = Set();
	int face, high, numBreaks = breaks.size();

	binop add = _add(addition, field);
	binop multiply = _mult(multiplication, field);
	unop negate = _neg(negation);
	unop invert = _inv(inversion);

	char q, r, s, inv, prod, result;

	for (int d = numBreaks-1; d > dimension-1; d--) {

		high = (d+1 >= numBreaks ? cellCount : breaks[d+1]);

		for (int j = breaks[d]; j < high; j++) {
			// If we're of the wrong dimension, keep going.
			// if (dim(j, breaks) != d) { continue; }
			cell = Boundary[j];

			while (!cell.empty() && nextColumnAdded[youngestOf(cell)] != 0) {
				// Get the "youngest" cell in the boundary and subtract it from
				// the current cell.
				youngest = Boundary[nextColumnAdded[youngestOf(cell)]];

				// Get the multiplicative inverse of the coefficient and do
				// arithmetic over the row.
				q = youngest[youngestOf(cell)];
				inv = invert(q);

				for (auto it=youngest.begin(); it != youngest.end(); ++it) {
					face = it->first;
					s = it->second;

					// Take the product of inv(q) with the entry of this row;
					// if this is row youngestOf(cell), then this product is 1
					// (it's a pivot). If the current cell shares this face, do
					// the subtraction; otherwise, just add a new coefficient.
					prod = multiply(inv, s);
					// prod = multiplication[inv][s];

					if (cell.count(face) > 0) {
						result = add(cell[face], negate(prod));
						// result = addition[cell[face]][negation[prod]];

						if (result < 1) {
							cell.erase(face);
						} else {
							cell[face] = result;
						}
					} else {
						cell[face] = negate(prod);
					}
				}
			}
			// Check whether we've eliminated the column. For some god damn reason
			// we have to re-set the entry of the Boundary??? Why?????? Scope??? wtf
			if (!cell.empty()) {
				Boundary[j] = cell;
				nextColumnAdded[youngestOf(cell)] = j;
				Boundary[youngestOf(cell)] = Column();
			} else {
				marked.insert(j);
			}
		}
	}

	// Find the essential birth times by checking whether the column is marked
	// (i.e. is a cycle) and has no younger columns to add. (For some reason,
	// 0 gets left out here. Not sure why...)
	Set essential = Set();
	essential.insert(0);

	for (auto it = marked.begin(); it != marked.end(); it++) {
		if (nextColumnAdded[*it] == 0) essential.insert(*it);
	}

	return essential;
}


Bases LinearComputeBases(
		int field, Lookup addition, Lookup multiplication, Lookup negation, Lookup inversion,
		BoundaryMatrix Boundary, Index breaks, int cellCount, int dimension
	) {
	Index nextColumnAdded = Index(cellCount, 0);
	Column cell, youngest;
	Set marked = Set();
	Map dimensions = Map();
	int face, high, numBreaks = breaks.size();

	// Keep track of the linear combinations used to reduce each column; these
	// give us (representatives of) the basis for each homology group.
	BoundaryMatrix reducedColumns = BoundaryMatrix(cellCount, Column());

	binop add = _add(addition, field);
	binop multiply = _mult(multiplication, field);
	unop negate = _neg(negation);
	unop invert = _inv(inversion);

	char q, r, s, inv, prod, result;

	for (int d = numBreaks-1; d > dimension-1; d--) {

		high = (d+1 >= numBreaks ? cellCount : breaks[d+1]);

		for (int j = breaks[d]; j < high; j++) {
			// If we're of the wrong dimension, keep going.
			// if (dim(j, breaks) != d) { continue; }
			cell = Boundary[j];
			Column reduced = Column();

			while (!cell.empty() && nextColumnAdded[youngestOf(cell)] != 0) {
				// Get the "youngest" cell in the boundary and subtract it from
				// the current cell.
				youngest = Boundary[nextColumnAdded[youngestOf(cell)]];

				// Get the multiplicative inverse of the coefficient and do
				// arithmetic over the row.
				q = youngest[youngestOf(cell)];
				inv = invert(q);

				// Given the coefficient on the pivot entry, mark the columns
				// used to eliminate this one.
				reduced[nextColumnAdded[youngestOf(cell)]] = negate(inv);

				for (auto it=youngest.begin(); it != youngest.end(); ++it) {
					face = it->first;
					s = it->second;

					// Take the product of inv(q) with the entry of this row;
					// if this is row youngestOf(cell), then this product is 1
					// (it's a pivot). If the current cell shares this face, do
					// the subtraction; otherwise, just add a new coefficient.
					prod = multiply(inv, s);

					if (cell.count(face) > 0) {
						result = add(cell[face], negate(prod));
						// result = addition[cell[face]][negation[prod]];

						if (result < 1) {
							cell.erase(face);
						} else {
							cell[face] = result;
						}
					} else {
						cell[face] = negate(prod);
					}
				}
			}
			// Check whether we've eliminated the column. For some god damn reason
			// we have to re-set the entry of the Boundary??? Why?????? Scope??? wtf
			if (!cell.empty()) {
				Boundary[j] = cell;
				nextColumnAdded[youngestOf(cell)] = j;
				Boundary[youngestOf(cell)] = Column();
			} else {
				marked.insert(j);
				dimensions[j] = d;
			}

			reducedColumns[j] = reduced;
		}
	}

	// Find the essential birth times by checking whether the column is marked
	// (i.e. is a cycle) and has no younger columns to add. (For some reason,
	// 0 gets left out here. Not sure why...)
	Bases bases = Bases(numBreaks, Basis());

	for (auto it = marked.begin(); it != marked.end(); it++) {
		if (nextColumnAdded[*it] == 0) {
			bases[dimensions[*it]].push_back(reducedColumns[*it]);
		}
	}

	return bases;
}
