#include "mlp.hpp"

using namespace std;

int main( void) {

	vector<int> neurons;
	neurons.push_back( 28 * 28);
	neurons.push_back( 256);
	neurons.push_back( 10);
	cMLP mlp( neurons);
	

	return 0;
}