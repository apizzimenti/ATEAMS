
#include <set>
#include <vector>
#include <map>
#include <iostream>

using namespace std;

// Type for indices and type for data; for now, we use unsigned 32-bit ints for
// indices (since we're never negative) and signed chars for data (since our
// fields are never that big).
typedef int32_t INDEXTYPE;
typedef char DATATYPE;

// Standard sets, indices
typedef set<INDEXTYPE> Set;
typedef map<INDEXTYPE,INDEXTYPE> Map;
typedef vector<INDEXTYPE> Index;

// Arithmetic.
typedef vector<DATATYPE> Lookup;
typedef vector<Lookup> Table;

// Persistence pairs; these are literally just pairs of ints.
typedef vector<Index> PersistencePairs;

// Matrices.
typedef map<INDEXTYPE,DATATYPE> Column;
typedef map<INDEXTYPE,Column> SparseLinearCombination;
typedef vector<Column> BoundaryMatrix;
typedef vector<Index> FlatBoundaryMatrix;


// Helper method for printing `map`s.'
template <typename MapStorage>
void printmap(MapStorage m) {
	for (const auto& [k, v] : m) {
		cout << (int)k << ": " << (int)v << endl;
	}
}

