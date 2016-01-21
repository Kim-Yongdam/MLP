#include "mlp.hpp"

using namespace std;
using namespace cv;


std::vector<double> activation_value;		//Opk
std::vector<double> derivation_value;		//delta pj
std::vector<double> activated_output;		//Opj
std::vector<double> derivated_output;		//delta pk


//http://www.aistudy.com/neural/MLP_kim.htm#_bookmark_165f298
//Multi-Layer Perceptron 참조 페이지

#define LEARNING_RATE 0.01
#define CLASS_SIZE 10
#define ITER_COUNT 1000

void init(){
	/*
	memset(activation_value, 0, sizeof(double) * 256);
	memset(derivation_value, 0, sizeof(double) * 256);
	memset(activated_output, 0, sizeof(double) * 10);
	memset(derivated_output, 0, sizeof(double) * 10);
	*/
	activation_value.resize(256);
	activated_output.resize(10);
	derivation_value.resize(256);
	derivated_output.resize(10);
}

double derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x) {

	return (1 - exp( -vector_multiplication(w,x)) / sqrt(1 + exp( -vector_multiplication(w,x))));
}


double derivate_relu( const std::vector<double>& w, const std::vector<double>& x) {

	double z = vector_multiplication(w, x);

	if(z > 0 ) return 1;
	else return 0;
}


// http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
double derivate_softmax( const std::vector<double>& w, const std::vector<double>& x) {
	//TO DO
	return 0;
}


double sigmoid( const std::vector<double>& w, const std::vector<double>& x) {

	return 1 / (1 + exp( -vector_multiplication(w, x)));
}


double relu( const std::vector<double>& w, const std::vector<double>& x) {

	return max(0., vector_multiplication(w,x));
}


double softmax( const std::vector<double>& w, const std::vector<double>& x) {

	return 0;
}


void cOutputLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++)
		output[i] = exp( vector_multiplication( w2[i], x));

	double sum = 0;
	for( int i = 0 ; i < w2.size() ; i++)
		sum += output[ i];

	for( int i = 0 ; i < w2.size() ; i++){
		output[ i] /= sum;
		activated_output[i] = output[i];	//굿
	}
}

void cOutputLayer::backward_prop( const std::vector<double>& x, const std::vector<double>& delta_p, std::vector<double>& output){

	//delta_p : 목표 출력값 (one-vs-all)

	output.resize( w2.size());
	for(int j = 0; j < output.size(); j++){
	//	output[j] = derivate_active_func(w2[j], x);
		derivated_output[j] = activated_output[j] - delta_p[j];	//굿
		for(int k = 0; k < CLASS_SIZE; k++)
			w2[j][k] = w2[j][k] + LEARNING_RATE * derivated_output[j] * activation_value[j]; //굿
	}
}


void cHiddenLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++){
		output[i] = active_func(w2[i], x);
		activation_value[i] = output[i]; //굿
	}
}

void cHiddenLayer::backward_prop( const std::vector<double>& x, const std::vector<double>& t_p, std::vector<double>& output){

	//t_p : 목표 출력값


	output.resize( w2.size());
	for(int i = 0; i < output.size(); i++){
		output[i] = derivate_active_func(w2[i], x);
		for(int j = 0; j < t_p.size(); j++) {
			derivation_value[i] = output[i] * derivated_output[i] * w2[i][j];	//굿
			w2[i][j] = w2[i][j] + LEARNING_RATE * derivation_value[i] * output[i];
		}
	}

	/*
	2016.01.20

	수정사항 : 
	1. weight 값을 w2로 부터 받아와야 한다. 함수원형을 수정하려 했으나, 선언문을 모두 고쳐야 해서 fail. w2 자체를 쓰는 방법을 찾아봐야 겠다.
	2. 6단계를 마친 뒤에, 7단계에서 weight와 theta를 어떻게 업데이트 할 것인가. //NN graph 개형 생각해서 값 대입.

	*/




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
		init();

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

		std::vector<int> pred_label = predict( data);
		cout << "iter : " << iter << ", accuracy : " << calcMNIST_test_error( data, pred_label) << endl;
		if(iter % 100 == 0) cout << "Now training.." << endl;
	}
}

std::vector<int> cMLP::predict( const std::vector< datum>& data) {

	std::vector<int> pred_label( data.size(), 0);

	for(int iter = 0; iter < data.size(); iter++){

		const auto& d = data[ iter];
		vector<double> output1 = d.x;
		for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++) {
			auto& layer = layers[ nlayer];

			vector<double> output2;
			layer->forward_prop( output1, output2);
			output1 = output2;
			
		}
		pred_label[iter] = getMaxInt(output1);
	}
	
	return pred_label;
}
/*
std::vector<int> predict( std::vector< datum>& data) {

std::vector<int> pred_label( data.size(), 0);
std::vector<double> pred_rate(10, 0);

for(int i = 0; i < data.size(); i++){
for(int j = 0; j < CLASS_SIZE; j++)
pred_rate[j] = exp( vector_multiplication(data[i].x, w[j]));
pred_label[i] = getMaxInt(pred_rate);
}

return pred_label;
};
*/