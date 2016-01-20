#include "mlp.hpp"

using namespace std;

int main( void) {

	vector<int> neurons;
	neurons.push_back( 28 * 28);
	neurons.push_back( 256);
	neurons.push_back( 10);
	cMLP mlp( neurons);

	vector< datum> train_data, test_data;
	getMNIST( "../../MNIST DB", train_data, 100, test_data, 0);
	
	std::for_each( train_data.begin(), train_data.end(), [] ( datum& val1) { 
		std::for_each( val1.x.begin(), val1.x.end(), [] ( double& val2) {
			val2 /= 255.; 
		});
	});

	mlp.train( train_data, 100, 0.01);

	return 0;
}