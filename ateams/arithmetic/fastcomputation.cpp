
#include <vector>
#include <givaro/modular.h>
#include <linbox/matrix/sparse-matrix.h>

using namespace std;
using namespace LinBox;

typedef Givaro::Modular<float> Field;

int kernel(vector<int> A, int rank, int nullity, int field) {
	// Construct the finite field and the sparse matrix of appropriate dimension.
	Field F((float)field);

	SparseMatrix<Field> B(F);
	return F.minElement();
}
