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


// http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
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


void cOutputLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++)
		output[i] = active_func(w2[i], x);

	double sum = 0;
	for( int i = 0 ; i < w2.size() ; i++)
		sum += output[ i];

	for( int i = 0 ; i < w2.size() ; i++)
		output[ i] /= sum;
}

void cOutputLayer::backward_prop( const std::vector<double>& x, const std::vector<double>& sigma_p, std::vector<double>& output){

	output.resize( w2.size());
	for(int i = 0; i < output.size(); i++){
		output[i] = derivate_active_func(w2[i], x);
	}
}


void cHiddenLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++)
		output[i] = active_func(w2[i], x);
}

void cHiddenLayer::backward_prop( const std::vector<double>& x, const std::vector<double>& t_p, std::vector<double>& output){

	output.resize( w2.size());
	for(int i = 0; i < output.size(); i++){
		output[i] = derivate_active_func(w2[i], x);
	}

	
	/*
	// 5단계  --------------------------------------------
	vector<double> sigma_p( w2.size(), 0);
	for( int nsigma = 0 ; nsigma < sigma_p.size() ; nsigma++)
		sigma_p[ nsigma] = ( output1[ nsigma] - t_p[ nsigma]) * derivate_active_func( t_p);
	// 5단계 끝===========================================

	// 6단계

	// 7단계
	*/
}

void cMLP::train( const std::vector< datum>& data, const int iteration, const double learning_rate) {

	for( int iter = 0 ; iter < iteration ; iter++) {
		// TO DO

		for( int nd = 0 ; nd < data.size() ; nd++) {
			const auto& datum = data[ nd];

			// 2단계 ~ 4 단계
			vector<double> output1 = datum.x;
			for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++) {
				auto& layer = layers[ nlayer];

				vector<double> output2;
				layer->forward_prop( output1, output2);
				output1 = output2;
			}

			
			//one hot encoding
			vector<double> t_p1( output1.size(), 0);
			t_p1[ datum.label] = 1;			
			for( int nlayer = layers.size() - 1 ; nlayer >= 0 ; nlayer--) {
				
				vector<double> t_p2;
				auto& layer = layers[ nlayer];
				layer->backward_prop( datum.x, t_p1, t_p2);
			}
		}
	}
}



std::vector<int> cMLP::predict( const std::vector< datum>& data) {

	std::vector<int> pred_label( data.size());

	// TO DO

	return pred_label;
}