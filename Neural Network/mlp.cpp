#include "mlp.hpp"

using namespace std;
using namespace cv;


#define LEARNING_RATE 0.01
#define CLASS_SIZE 10
#define ITER_COUNT 1000


double derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x) {
	// TO DO
	return (1 - exp( -vector_multiplication(w,x)) / sqrt(1 + exp( -vector_multiplication(w,x))));
}


double derivate_relu( const std::vector<double>& w, const std::vector<double>& x) {
	// TO DO
	// ver 111
	return max(0., vector_multiplication(w,x));
}


double derivate_softmax( const std::vector<double>& w, const std::vector<double>& x) {
	// TO DO
	return exp(vector_multiplication(w,x));
}


double sigmoid( const std::vector<double>& w, const std::vector<double>& x) {
	// TO DO

	return 1 / (1 + exp( -vector_multiplication(w, x)));
}


double relu( const std::vector<double>& w, const std::vector<double>& x) {
	// TO DO
	return max(0., vector_multiplication(w,x));
}


double softmax( const std::vector<double>& w, const std::vector<double>& x) {
	// TO DO
	return exp(vector_multiplication(w, x));
}

void cLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {
	for(int i =0; i < w2.size(); i++)
		output[i] = active_func(w2[i], x);
}

void cLayer::backward_prop( const std::vector<double>& x, std::vector<double>& output){
	for(int i = 0; i < output.size(); i++){
		output[i] = derivate_active_func(w2[i], x);


	}

}

void cMLP::train( const std::vector< datum>& data, const int iteration, const double learning_rate) {

	for( int iter = 0 ; iter < iteration ; iter++) {
		// TO DO
	}
}



std::vector<int> cMLP::predict( const std::vector< datum>& data) {

	std::vector<int> pred_label( data.size());

	// TO DO

	return pred_label;
}