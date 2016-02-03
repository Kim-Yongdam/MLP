#include "mlp.hpp"

using namespace std;

int main( void) {

	/*
	vector<int> neurons;
	neurons.push_back( 28 * 28);
	neurons.push_back( 256);
	neurons.push_back( 10);
	cMLP mlp( neurons);
	*/

	vector< sLayer_shape> l_shapes;
	l_shapes.push_back( sLayer_shape( 28 * 28, 256, 0.5));
	l_shapes.push_back( sLayer_shape( 256, 10, 0.8));
	cMLP mlp( l_shapes);

	vector< datum> train_data, test_data;

	int train_data_size = 60000;
	int test_data_size = 10000;
	
	
	getMNIST( "../../DB/MNIST/", train_data, train_data_size, test_data, test_data_size);
	std::for_each( train_data.begin(), train_data.end(), [] ( datum& val1) { 
		std::for_each( val1.x.begin(), val1.x.end(), [] ( double& val2) {
			val2 /= 255.; 
		});
	});

	std::random_shuffle( train_data.begin(), train_data.end());
	
	//mlp.loadModel( "model");
	mlp.train( train_data, 10000, 0.7, 10000, 0.9, 0, 0.01);


	const auto& first_layer_w2 = mlp.getLayer()[ 0]->getW2();
	for( int noutput = 0 ; noutput < first_layer_w2.size() ; noutput++) {
		const auto& w = first_layer_w2[ noutput];

		cv::Mat img = cv::Mat( 28, 28, CV_32FC1);
		for( int ninput = 0 ; ninput < w.size() - 1 ; ninput++) {
			img.at<float>( ninput) = w[ ninput];
		}

		cv::normalize( img, img, 0, 1, CV_MINMAX);
		cv::imshow( std::to_string( noutput), img);
	}
	cv::waitKey();

	double result;

	std::vector<int> pred_label = mlp.predict(test_data);
	result = calcMNIST_test_error(test_data, pred_label);

	printf("Matching Rate : %.2f\n", result);

	return 0;
}