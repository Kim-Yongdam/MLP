#include "mlp.hpp"

using namespace std;

int main( void) {


	vector<int> neurons;
	neurons.push_back( 28 * 28);
	neurons.push_back( 256);
	neurons.push_back( 10);
	cMLP mlp( neurons);

	vector< datum> train_data, test_data;
	getMNIST( "../../MNIST DB", train_data, 60000, test_data, 0);
	std::for_each( train_data.begin(), train_data.end(), [] ( datum& val1) { 
		std::for_each( val1.x.begin(), val1.x.end(), [] ( double& val2) {
			val2 /= 255.; 
		});
	});

	std::random_shuffle( train_data.begin(), train_data.end());
	mlp.train( train_data, 10, 0.3);

	double result;

	std::vector<int> pred_label = mlp.predict(test_data);
	result = calcMNIST_test_error(test_data, pred_label);

	printf("Matching Rate : %.2f\n", result);

	return 0;
}