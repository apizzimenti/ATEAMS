
#include <phat/compute_persistence_pairs.h>
#include <phat/boundary_matrix.h>
#include <phat/representations/default_representations.h>
#include <phat/algorithms/twist_reduction.h>
#include <phat/algorithms/standard_reduction.h>
#include <phat/algorithms/row_reduction.h>
#include <phat/algorithms/chunk_reduction.h>
#include <phat/algorithms/spectral_sequence_reduction.h>
#include <phat/algorithms/swap_twist_reduction.h>
#include <phat/algorithms/exhaustive_compress_reduction.h>
#include <phat/algorithms/lazy_retrospective_reduction.h>
#include <phat/helpers/dualize.h>

#include "Persistence.h"

using namespace std;

/*
#################################
##### PERSISTENCE WITH PHAT #####
#################################
*/
typedef vector<phat::index> PHATColumn;
typedef phat::boundary_matrix<phat::bit_tree_pivot_column> PHATBoundaryMatrix;
typedef phat::persistence_pairs Pairs;
typedef phat::twist_reduction Twist;


Index ReindexBoundaryMatrix(Index &boundary, Index filtration, int homology, Index breaks) {
	// Decide when we should be editing rows or columns.
	int cellCount = filtration.size();
	int low = breaks[homology];
	int high = breaks[homology+1];
	int higher = (homology+2 >= breaks.size() ? cellCount : breaks[homology+2]);
	int row, column;

	// Reindex the boundary matrix in-place so we don't have to do it in Python.
	// Eugh.
	Map IndexMap = Map();

	for (int t=low; t < higher; t++) {
		IndexMap[filtration[t]] = t;
	}

	for (int i=0; i < boundary.size(); i+=3) {
		row = boundary[i];
		column = boundary[i+1];

		if (low <= column && column < high) {
			// In this situation, we're editing the *column* --- for example,
			// if the (usual) 9th element was placed 10th, then we have to change
			// the 9th column to the 10th one.
			boundary[i+1] = IndexMap[column];
		} else if (high <= column && column < higher) {
			// Here, we're editing the *row* --- since we're in a higher-dimensional
			// cell, we need to point toward the correct (re-indexed) row.
			boundary[i] = IndexMap[row];
		}
	}
	
	return boundary;
}


FlatBoundaryMatrix Flatten(Index full, int columns, Index breaks) {
	FlatBoundaryMatrix boundary = FlatBoundaryMatrix(columns, Index());
	int row, column;

	for (int t=0; t < full.size(); t+=3) {
		row = full[t];
		column = full[t+1];

		// If the column corresponds to a cell that isn't a vertex, we insert the
		// row into the column. Afterwards, we'll have to go through and sort
		// each of the columns as well.
		if (breaks[1] <= column) { boundary[column].push_back(row); }
	}

	// Sort column entries (presumably so this makes building the bitpacks easier
	// later?).
	for (int i=0; i < columns; i++) {
		sort(boundary[i].begin(), boundary[i].end());
	}

	return boundary;
}


FlatBoundaryMatrix ReindexAndFlatten(Index _boundary, Index filtration, int homology, Index breaks) {
	_boundary = ReindexBoundaryMatrix(_boundary, filtration, homology, breaks);
	return Flatten(_boundary, filtration.size(), breaks);
}


PersistencePairs PHATComputePersistencePairs(Index _boundary, Index filtration, int homology, Index breaks) {
	// Build out the boundary matrix.
	Pairs pairs;
	FlatBoundaryMatrix boundary = ReindexAndFlatten(_boundary, filtration, homology, breaks);
	PHATBoundaryMatrix Boundary;
	PHATColumn faces;
	int numFaces, numPHATColumns = boundary.size();

	// Count the number of columns.
	Boundary.set_num_cols(numPHATColumns);

	for (int t=0; t < numPHATColumns; t++) {
		// Fill in the faces for this column. Should be fine dealing with empty
		// columns (i.e. vertices).
		numFaces = boundary[t].size();

		for (int j=0; j < numFaces; j++) { faces.push_back(boundary[t][j]); }

		Boundary.set_dim(t, numFaces*2);
		Boundary.set_col(t, faces);
		faces.clear();
	}

	// Compute the persistence pairs and populate a Vector to return to the
	// user.
	phat::compute_persistence_pairs<Twist>(pairs, Boundary);
	pairs.sort();

	PersistencePairs Pairs(pairs.get_num_pairs(), Index(2));

	for (phat::index i=0; i < pairs.get_num_pairs(); i++) {
		Pairs[i][0] = pairs.get_pair(i).first;
		Pairs[i][1] = pairs.get_pair(i).second;
	}

	return Pairs;
}


/*
#######################
##### PERSISTENCE #####
#######################
*/

#include <functional>


template <typename BalancedStorage>
int youngestOf(BalancedStorage column) {
	// Gets the "youngest" (largest-indexed) cell in the column.
	return column.rbegin()->first;
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
			// Make sure we include *all* the coefficients!
			reducedColumns[*it][*it] = (DATATYPE)1;
			bases[dimensions[*it]].push_back(reducedColumns[*it]);
		}
	}

	return bases;
}


/*
##################################
##### PERSISTENCE WITH SPASM #####
##################################
*/

#include <cmath>
#include <format>

extern "C" {
	#include <spasm/spasm.h>
}

typedef struct spasm_csr Matrix;
typedef struct spasm_triplet Triplet;
typedef struct spasm_lu Echelonized;
typedef spasm_ZZp Zp;


Matrix *SparseMatrixFill(Basis B, int M, int N, int characteristic, int augment=0) {
	Triplet *T = spasm_triplet_alloc(M, N, 1, characteristic, true);

	cerr << "[Persistence] building sparse coboundary matrix...";
	for (int col=augment; col<B.size(); col++) {
		for (auto const& [row, q] : B[col]) {
			// Insert them into the matrix.
			spasm_add_entry(T, row, col, q);
		}
	}

	Matrix *A = spasm_compress(T);
	spasm_triplet_free(T);
	cerr << " done." << endl;
	return A;
}


Matrix *SparseSubmatrixFill(BoundaryMatrix B, int r0, int r1, int c0, int c1, int characteristic) {
	Triplet *T = spasm_triplet_alloc(r1-r0, c1-c0, 1, characteristic, true);
	
	cerr << "[Persistence] building sparse submatrix...";
	for (int col=c0; col<B.size(); col++) {
		for (auto const& [row,q] : B[col]) {
			if (r0 <= row && row < r1) spasm_add_entry(T, row-r0, col-c0, q);
		}
	}

	Matrix *A = spasm_compress(T);
	spasm_triplet_free(T);
	cerr << " done." << endl;
	return A;
}


Matrix *IdentityOfSize(int M, int N, int characteristic) {
	Triplet *T = spasm_triplet_alloc(M, N, 1, characteristic, true);

	cerr << "[Persistence] building cobasis system... ";
	for (int r=0; r<M; r++) spasm_add_entry(T, r, r, 1);

	Matrix *A = spasm_compress(T);
	spasm_triplet_free(T);
	cerr << "done." << endl;
	return A;
}


Basis ComputeCobasis(
	Basis combined, int M, int N, int rank, int characteristic, bool verbose
) {
	int _supp;
	if (!verbose) _supp = _suppress();

	// Construct the augmented coboundary.
	Matrix *B = SparseMatrixFill(combined, M, N, characteristic);
	Matrix *images = IdentityOfSize(rank, N, characteristic);

	struct echelonize_opts opts;
	spasm_echelonize_init_opts(&opts);
	opts.L = 1;

	// Solve.
	cerr << "[Persistence] echelonizing... ";
	Echelonized *ech = spasm_echelonize(B, &opts);
	cerr << "done." << endl;
	cerr << "[Persistence] solving... ";
	// bool *exists = (bool *)spasm_malloc(rank*sizeof(*exists));
	Matrix *solutions = spasm_gesv(ech, images, NULL);
	cerr << "done." << endl;
	cerr << "[Persistence] building the cobasis... ";

	// Report the cobasis.
	Basis cobasis(rank, Column());

	const int *columnIndex = solutions->j;
	const i64 *rowIndex = solutions->p;
	const Zp *values = solutions->x; 

	for (int i=0; i<rank; i++) {
		for (int px=rowIndex[i]; px<rowIndex[i+1]; px++) {
			int q = (values[px]+characteristic)%characteristic;
			cobasis[i][columnIndex[px]] = q;
		}
	}

	cerr << "done." << endl;

	// Cleanup...
	spasm_csr_free(B);
	spasm_csr_free(images);
	spasm_csr_free(solutions);
	spasm_lu_free(ech);
	// free((int*)columnIndex);
	// free((i64*)rowIndex);
	// free((Zp*)values);
	// free(exists);
	if (!verbose) _resume(_supp);

	return cobasis;
}


Set SolveComputePercolationEvents(BoundaryMatrix boundary, Basis _cobasis, int M, int N, int rank, int characteristic) {
	Matrix *coboundaryT = SparseMatrixFill(boundary, M, N, characteristic);
	Matrix *cobasis = SparseMatrixFill(_cobasis, M, rank, characteristic);
	Matrix *cobasisT = spasm_transpose(cobasis, 1);
	int _supp = _suppress();

	struct echelonize_opts opts;
	spasm_echelonize_init_opts(&opts);
	opts.L = 1;

	for (int t=1; t<N; t++) {
		// Take submatrices.
		Matrix *subcoboundaryT = spasm_submatrix(coboundaryT, 0, M, 0, t, 1);
		Matrix *subcobasisT = spasm_submatrix(cobasisT, 0, rank, 0, t, 1);

		printSpaSMmat<Matrix, Zp, i64>(subcoboundaryT);
		printSpaSMmat<Matrix, Zp, i64>(subcobasisT);

		// Solve; check whether a solution exists (i.e. whether any of the giant
		// cocycles are in the image).
		Echelonized *subcoboundaryE = spasm_echelonize(subcoboundaryT, &opts);

		bool *exists = (bool *)spasm_malloc(subcobasisT->n*sizeof(*exists));
		Matrix *solutions = spasm_gesv(subcoboundaryE, subcobasisT, exists);

		cerr << t << endl;
		for (int i=0; i<rank; i++) cerr << exists[i] << " ";
		cerr << endl;
	}
	_resume(_supp);

	return Set();
}


void _topologicalOrder(Index ordered, Index &top, int l, int r, int* p) {
	int t = std::floor((r+l)/2);
	top[*p] = t;
	*p += 1;

	// Branch or inner leaf vertices
	if (2 < r-l) {
		if (1 < t-l) _topologicalOrder(ordered, top, l, t, p);
		if (1 < r-t) _topologicalOrder(ordered, top, t, r, p);
	}
}

Index topologicalOrder(int rank) {
	Index ordered(rank+1), top(rank+1);
	for (int i=0; i<rank+1; i++) ordered[i] = i;

	int p = 0;

	_topologicalOrder(ordered, top, -1, rank+1, &p);

	return top;
}


/*
	OVERLOAD: finds all percolation events.
*/
Set RankComputePercolationEvents(
	BoundaryMatrix augmented, int M, int N, int rank, int characteristic, bool verbose
) {
	// Verbose output if we're debugging.
	int _supp;
	if (!verbose) _supp = _suppress();

	// Fill the full matrix (which includes the cobasis); reserve names for
	// echelonizing.
	Matrix *full = SparseMatrixFill(augmented, M, N, characteristic);

	// Matrix options?
	struct echelonize_opts opts;
	spasm_echelonize_init_opts(&opts);
	// opts.L = 1;
	opts.enable_tall_and_skinny = 1;

	// Binary search setup.
	Map eventMarkers;
	Set events;
	Index top = topologicalOrder(rank), ranks(2);
	int withRank, noRank, LEFT, RIGHT, stop, t;

	// The homology group has rank 0 when we begin; no need to search for that.
	eventMarkers[0] = 0;

	for (int i=0; i<rank+1; i++) {
		// Find the largest and smallest indices less than and greater than
		// the current one; these specify the search window. We don't care about
		// when it becomes 0-dimensional though, so skip that one (it's already
		// marked).
		LEFT = 0, RIGHT = N, stop = top[i];
		if (stop == 0) continue;

		for (auto const& [rnk, ind] : eventMarkers) {
			if (rnk < stop) LEFT = ind;
			if (rnk > stop && ind < RIGHT) RIGHT = ind;
		}
		
		// Binary search between the wickets.
		while (LEFT <= RIGHT) {
			t = LEFT + std::floor((RIGHT-LEFT)/2);

			for (int s=t; s<t+2; s++) {
				// Take submatrices. If we're short and wide, take the transpose
				// before echelonizing.
				Matrix *withBasis, *noBasis;

				if (M < s) {
					const Matrix *wB = spasm_submatrix(full, 0, M, 0, s, 1);
					withBasis = spasm_transpose(wB, 1);

					const Matrix *nB = spasm_submatrix(full, rank, M, 0, s, 1);
					noBasis = spasm_transpose(nB, 1);

					spasm_csr_free((Matrix *)wB);
					spasm_csr_free((Matrix *)nB);
				} else {
					withBasis = spasm_submatrix(full, 0, M, 0, s, 1);
					noBasis = spasm_submatrix(withBasis, rank, M, 0, s, 1);
				}

				// withBasis = spasm_submatrix(full, 0, M, 0, s, 1);
				// noBasis = spasm_submatrix(withBasis, rank, M, 0, s, 1);

				// Echelonize, get the ranks, mark the ranks.
				cerr << "[Persistence] found submatrices, echelonizing with basis... " << endl;
				Echelonized *withBasisE = spasm_echelonize(withBasis, &opts);
				cerr << "[Persistence] done, echelonizing without basis... " << endl;
				Echelonized *noBasisE = spasm_echelonize(noBasis, &opts);
				cerr << "[Persistence] done. computing ranks..." << endl;

				withRank = withBasisE->U->n;
				noRank = noBasisE->U->n;
				ranks[s-t] = withRank-noRank;

				spasm_csr_free(withBasis);
				spasm_csr_free(noBasis);
				spasm_lu_free(withBasisE);
				spasm_lu_free(noBasisE);
			}

			// If the difference between the computed ranks is 1, then we're done;
			// we've found our spot to break. Otherwise, continue.
			int r1 = ranks[0], r2 = ranks[1];
			bool samerank = (r1 == r2);

			// If r1 > r2, then we have a problem.
			if (r1 > r2) goto cleanup;

			if (samerank) {
				cerr << format("[Persistence] same rank {} == {}, ", r1, r2);
				if (r1 < stop) {
					LEFT = t+1;
					cerr << "[Persistence] moving forward" << endl;
				} else {
					RIGHT = t-1;
					cerr << "[Persistence] moving backward" << endl;
				}
			} else {
				cerr << format("[Persistence] different ranks {} != {}, ", r1, r2);
				if (r2 == stop) {
					cerr << "right rank, exiting" << endl;
					goto found;
				} else {
					cerr << "wrong rank, ";
					if (r2 < stop) {
						LEFT = t+1;
						cerr << "moving forward" << endl;
					} else {
						RIGHT = t-1;
						cerr << "moving backward" << endl;
					}
				}
			}
		}
		found:
			eventMarkers[stop] = t;
			events.insert(t);

			if (verbose) {
				cerr << "[Persistence] marked events are ";
				printmap<Map>(eventMarkers);
				cerr << endl;
				cerr << endl;
			}
	}

	cleanup:
		spasm_csr_free(full);
		// spasm_csr_free(withBasis);
		// spasm_csr_free(noBasis);
		// spasm_lu_free(withBasisE);
		// spasm_lu_free(noBasisE);

		if (!verbose) _resume(_supp);

	return events;
}

/*
	OVERLOAD: stops at a specified event, doesn't look for all of them.
*/
Set RankComputePercolationEvents(
	BoundaryMatrix augmented, int M, int N, int rank, int characteristic, int stop, bool verbose
) {
	// Verbose output if we're debugging.
	int _supp;
	if (!verbose) _supp = _suppress();

	// Matrix options?
	struct echelonize_opts opts;
	spasm_echelonize_init_opts(&opts);
	// opts.L = 1;
	opts.enable_tall_and_skinny = 1;

	// We don't do binary search with this method, we just find the middle event.
	// Start searching around the expected percolation probability.
	Set events;
	Index ranks(2);
	int withRank, noRank, LEFT = 0, RIGHT = N, t = (int)(N*sqrt(characteristic)/(1+sqrt(characteristic)));

	// Binary search between the wickets.
	while (LEFT <= RIGHT) {

		for (int s=t; s<t+2; s++) {
			// Take submatrices.
			Matrix *withBasis = SparseSubmatrixFill(augmented, 0, M, 0, s, characteristic);
			Matrix *noBasis = SparseSubmatrixFill(augmented, rank, M, 0, s, characteristic);

			// Echelonize, get the ranks, mark the ranks.
			cerr << "[Persistence] found submatrices, echelonizing with basis... " << endl;
			Echelonized *withBasisE = spasm_echelonize(withBasis, &opts);
			cerr << "[Persistence] done, echelonizing without basis... " << endl;
			Echelonized *noBasisE = spasm_echelonize(noBasis, &opts);
			cerr << "[Persistence] done. computing ranks..." << endl;

			withRank = withBasisE->U->n;
			noRank = noBasisE->U->n;
			ranks[s-t] = withRank-noRank;

			spasm_csr_free(withBasis);
			spasm_csr_free(noBasis);
			spasm_lu_free(withBasisE);
			spasm_lu_free(noBasisE);
		}

		// If the difference between the computed ranks is 1, then we're done;
		// we've found our spot to break. Otherwise, continue.
		int r1 = ranks[0], r2 = ranks[1];
		bool samerank = (r1 == r2);

		// If r1 > r2, then we have a problem.
		if (r1 > r2) goto cleanup;

		if (samerank) {
			cerr << format("[Persistence] same rank {} == {}, ", r1, r2);
			if (r1 < stop) {
				LEFT = t+1;
				cerr << "[Persistence] moving forward" << endl;
			} else {
				RIGHT = t-1;
				cerr << "[Persistence] moving backward" << endl;
			}
		} else {
			cerr << format("[Persistence] different ranks {} != {}, ", r1, r2);
			if (r2 == stop) {
				cerr << "right rank, exiting" << endl;
				goto found;
			} else {
				cerr << "wrong rank, ";
				if (r2 < stop) {
					LEFT = t+1;
					cerr << "moving forward" << endl;
				} else {
					RIGHT = t-1;
					cerr << "moving backward" << endl;
				}
			}
		}
		// Update the search location.
		t = LEFT + std::floor((RIGHT-LEFT)/2);
	}
	found:
		events.insert(t);

	cleanup:
		if (!verbose) _resume(_supp);

	return events;
}


/*
#######################################
##### PERSISTENCE WITH SPARSERREF #####
#######################################
*/

#include <SparseRREF/sparse_mat.h>
#include "util.h"

// #define USE_MIMALLOC 1

using namespace SparseRREF;

typedef sparse_mat<data_t, index_t> ZpZMatrix;
typedef sparse_vec<data_t, index_t> ZpZVector;
typedef field_t ZpZ;


ZpZMatrix subMatrixT(BoundaryMatrix B, int r0, int r1, int c0, int c1) {
	ZpZMatrix A(r1-r0, c1-c0);

	for (int col=c0; col<c1; col++) {
		for (auto const& [row, q] : B[col]) {
			if (r0 <= row && row < r1) {
				ZpZVector &matrow = A.rows[row-r0];
				matrow.push_back((index_t)(col-c0), (data_t)q);
			}
		}
	}

	A.compress();
	return A;
}


Set SRankComputePercolationEvents(BoundaryMatrix augmented, int M, int N, int rank, int characteristic, bool verbose) {
	int _supp;
	if (!verbose) _supp = _suppress();

	const ZpZ GFp(FIELD_Fp, characteristic);

	// Specify the order in we'll encounter each index.
	int withRank, noRank, LEFT, RIGHT, stop, t;
	Index top = topologicalOrder(rank), ranks(2);
	Map eventMarkers;
	Set events;

	eventMarkers[0] = 0;

	for (int i=0; i<rank+1; i++) {
		// Find the largest and smallest indices less than and greater than
		// the current one; these specify the search window. We don't care about
		// when it becomes 0-dimensional though, so skip that one (it's already
		// marked).
		LEFT = 0;
		RIGHT = N;
		stop = top[i];
		
		if (stop == 0) continue;

		for (auto const& [rnk, ind] : eventMarkers) {
			if (rnk < stop) LEFT = ind;
			if (rnk > stop && ind < RIGHT) RIGHT = ind;
		}
		
		// Binary search between the wickets.
		while (LEFT <= RIGHT) {
			t = LEFT + std::floor((RIGHT-LEFT)/2);

			for (int s=t; s<t+2; s++) {
				Flint::set_memory_functions();
				rref_option_t opt;
				opt->pool.reset();
				opt->verbose = verbose;

				// Take submatrices.
				cerr << format("[Persistence] [{}/{}] filling submatrices... ", s, N);
				ZpZMatrix withBasis = subMatrixT(augmented, 0, M, 0, s);
				ZpZMatrix noBasis(withBasis);

				// Clear the first `rank` rows of `withBasis` to remove the basis
				// instead of filling the whole matrix again.
				for (int _b=0; _b<rank; _b++) noBasis[_b].zero();
				noBasis.clear_zero_row();
				noBasis.compress();
				
				cerr << "done." << endl;

				// Echelonize, then find the ranks.
				vector<vector<pivot_t<index_t>>> withPivots;
				vector<vector<pivot_t<index_t>>> noPivots;

				cerr << format("[Persistence] [{}/{}] RREFing submatrices... ", s, N);
				withPivots = sparse_mat_rref<data_t, index_t>(withBasis, GFp, opt);
				cerr << "finished with pivots, resetting pool, reducing without pivots... ";
				noPivots = sparse_mat_rref<data_t, index_t>(noBasis, GFp, opt);
				cerr << " done." << endl;

				withRank = 0;
				noRank = 0;

				for (auto &wp : withPivots) withRank += wp.size();
				for (auto &np : noPivots) noRank += np.size();

				ranks[s-t] = withRank-noRank;
			}

			// If the difference between the computed ranks is 1, then we're done;
			// we've found our spot to break. Otherwise, continue.
			int r1 = ranks[0], r2 = ranks[1];
			bool samerank = (r1 == r2);

			// If r1 > r2, then we have a problem.
			if (r1 > r2) goto cleanup;

			if (samerank) {
				cerr << format("[Persistence] same rank {} == {}, ", r1, r2);
				if (r1 < stop) {
					LEFT = t+1;
					cerr << "moving forward" << endl;
				} else {
					RIGHT = t-1;
					cerr << "moving backward" << endl;
				}
			} else {
				cerr << format("[Persistence] different ranks {} != {}, ", r1, r2);
				if (r2 == stop) {
					cerr << "right rank, exiting" << endl;
					goto found;
				} else {
					cerr << "wrong rank, ";
					if (r2 < stop) {
						LEFT = t+1;
						cerr << "moving forward" << endl;
					} else {
						RIGHT = t-1;
						cerr << "moving backward" << endl;
					}
				}
			}
		}

		found:
			eventMarkers[stop] = t;
			events.insert(t);
			cerr << format("[Persistence] [{}/{}] found {}/{} percolation events.", t, N, events.size(), rank) << endl;
	}

	cleanup:
		if (!verbose) _resume(_supp);
		Flint::clear_cache();

	return events;
}


Set SRankComputePercolationEvents(BoundaryMatrix augmented, int M, int N, int rank, int characteristic, int stop, bool verbose) {
	int _supp;
	if (!verbose) _supp = _suppress();

	const ZpZ GFp(FIELD_Fp, characteristic);

	Flint::set_memory_functions();
	rref_option_t opt;
	opt->pool.reset();
	opt->verbose = verbose;

	// Specify the order in we'll encounter each index.
	int withRank, noRank, t, LEFT = 0, RIGHT = N;
	Set events;
	Index ranks(2);

	while (LEFT <= RIGHT) {
		t = LEFT + std::floor((RIGHT-LEFT)/2);

		for (int s=t; s<t+2; s++) {

			// Take submatrices.
			cerr << format("[Persistence] [{}/{}] filling submatrices... ", s, N);
			ZpZMatrix withBasis = subMatrixT(augmented, 0, M, 0, s);
			ZpZMatrix noBasis(withBasis);
			// ZpZMatrix noBasis = subMatrixT(augmented, rank, M, 0, s);

			// Clear the first `rank` rows of `withBasis` to remove the basis
			// instead of filling the whole matrix again.
			for (int _b=0; _b<rank; _b++) noBasis[_b].zero();
			noBasis.clear_zero_row();
			noBasis.compress();

			cerr << "done." << endl;

			// Echelonize, then find the ranks.
			vector<vector<pivot_t<index_t>>> withPivots;
			vector<vector<pivot_t<index_t>>> noPivots;

			cerr << format("[Persistence] [{}/{}] RREFing submatrices... ", s, N) << endl;
			withPivots = sparse_mat_rref<data_t, index_t>(withBasis, GFp, opt);
			cerr << "finished with pivots, resetting pool, reducing without pivots... ";
			noPivots = sparse_mat_rref<data_t, index_t>(noBasis, GFp, opt);
			cerr << "[Persistence] done RREFing." << endl;

			withRank = 0;
			noRank = 0;

			for (auto &wp : withPivots) withRank += wp.size();
			for (auto &np : noPivots) noRank += np.size();

			ranks[s-t] = withRank-noRank;
		}

		// If the difference between the computed ranks is 1, then we're done;
		// we've found our spot to break. Otherwise, continue.
		int r1 = ranks[0], r2 = ranks[1];
		bool samerank = (r1 == r2);

		// If r1 > r2, then we have a problem.
		if (r1 > r2) goto cleanup;

		if (samerank) {
			cerr << format("[Persistence] same rank {} == {}, ", r1, r2);
			if (r1 < stop) {
				LEFT = t+1;
				cerr << "moving forward" << endl;
			} else {
				RIGHT = t-1;
				cerr << "moving backward" << endl;
			}
		} else {
			cerr << format("[Persistence] different ranks {} != {}, ", r1, r2);
			if (r2 == stop) {
				cerr << "right rank, exiting" << endl;
				goto found;
			} else {
				cerr << "wrong rank, ";
				if (r2 < stop) {
					LEFT = t+1;
					cerr << "moving forward" << endl;
				} else {
					RIGHT = t-1;
					cerr << "moving backward" << endl;
				}
			}
		}
	}

	found:
		events.insert(t);
		cerr << format("[Persistence] [{}/{}] found {}/{} percolation events.", t, N, events.size(), rank) << endl;

	cleanup:
		if (!verbose) _resume(_supp);
		opt->abort=true;
		Flint::clear_cache();

	return events;
}
