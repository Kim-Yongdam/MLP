#include "mlp.hpp"

using namespace std;
using namespace cv;


//http://www.aistudy.com/neural/MLP_kim.htm#_bookmark_165f298
//Multi-Layer Perceptron 참조 페이지

#define LEARNING_RATE 0.1
#define CLASS_SIZE 10
#define ITER_COUNT 1000


double derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x) {

	//return (1 - exp( -vector_multiplication(w,x)) / sqrt(1 + exp( -vector_multiplication(w,x))));
	double val = 1 / (1 + exp( -vector_multiplication(w, x)));
	return val * ( 1 - val);
}


double derivate_relu( const std::vector<double>& w, const std::vector<double>& x) {

	double z = vector_multiplication(w, x);

	if(z > 0 ) return 1;
	else return 0;
}


double sigmoid( const std::vector<double>& w, const std::vector<double>& x) {

	return 1 / (1 + exp( -vector_multiplication(w, x)));
}


double relu( const std::vector<double>& w, const std::vector<double>& x) {

	return max(0., vector_multiplication(w,x));
}


double sigmoid_val( double val) {
	return 1 / ( 1 + exp( -val));
}


double derivate_sigmoid_val( double val) {
	return sigmoid_val( val) * ( 1 - sigmoid_val( val));
}



void cSoftMaxLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++)
		output[i] = exp( vector_multiplication( w2[i], x));

	double sum = 0;
	for( int i = 0 ; i < w2.size() ; i++)
		sum += output[ i];

	for( int i = 0 ; i < w2.size() ; i++){
		output[ i] /= sum;
	}
}


void cHiddenLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++)
		output[i] = active_func(w2[i], x);
}


// back-prop
void cMLP::train( const std::vector< datum>& data, const int iteration, const double learning_rate) {

	for( int iter = 0 ; iter < iteration ; iter++) {

		for( int nd = 0 ; nd < data.size() ; nd++) {
			const auto& d = data[ nd];


			vector< vector<double>> save_o_pk( layers.size() + 1); // 입력층 + ( 히든 + 출력)레이어 해서 layers.size() + 1
			vector<double> output1 = d.x;
			save_o_pk[ 0].insert( save_o_pk[ 0].end(), output1.begin(), output1.end());
			save_o_pk[ 0].push_back( 1);
			for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++) {
				auto& layer = layers[ nlayer];

				vector<double> output2;
				const auto& w2 = layer->getW2();
				output2.resize( w2.size());
				for(int i =0; i < w2.size(); i++)
					output2[i] = layer->active_func( w2[i], output1);

				output1 = output2;
				save_o_pk[ nlayer + 1].insert( save_o_pk[ nlayer + 1].end(), output1.begin(), output1.end());
				save_o_pk[ nlayer + 1].push_back( 1);
			}

			// 5단계 -------------------------------------------------------------------------------------
			//one hot encoding
			vector<double> t_p1( output1.size(), 0);
			t_p1[ d.label] = 1;			// 실제출력 값

			vector<double> delta_p_k1( output1.size());
			for( int ndelta = 0 ; ndelta < delta_p_k1.size() ; ndelta++)
				delta_p_k1[ ndelta] = ( t_p1[ ndelta] - output1[ ndelta]) * output1[ ndelta] * ( 1 - output1[ ndelta]);
			// 5단계 끝====================================================================================


			// 6단계 ~ 8단계
			for( int nlayer = layers.size() - 1 ; nlayer >= 0 ; nlayer--) {

				auto& layer = layers[ nlayer];
				auto& w2 = layer->getW2();
				vector<double> delta_p_j(257);
				int current_layer_size = save_o_pk[nlayer].size();
				int previous_layer_size = save_o_pk[nlayer+1].size();
				//delta_p_j는 이전 레이어의 error값인 delta_p_k1과 O_pj(save_o_pk의 두번째 레이어)으로 계산한다.
				/*
				delta_pj = calculated error value in current layer
				delta_pk = delta_p_k1[k] and also means propagated error value from previous layer
				Wkj = w2[k][j]
				Opj = save_o_pk[nlayer]
				theta_k = w2[k][save_o_pk[nlayer.size()]
				alpha & beta = learning_rate
				*/
				for(int j = 0; j < current_layer_size; j++) {
					for(int k = 0; k < previous_layer_size - 1; k++) {
						if(nlayer == 1) delta_p_j[j] += delta_p_k1[k] * (w2[k][j] * save_o_pk[nlayer][j]) * (1 - save_o_pk[nlayer][j]);
						w2[k][j] = w2[k][j] + (learning_rate * delta_p_k1[k] * save_o_pk[nlayer][j]);
						w2[k][current_layer_size - 1] = w2[k][current_layer_size-1] + learning_rate * delta_p_k1[k];
					}
				}
				// delta_p_j 구해서 이전 레이어로 전파
				delta_p_k1 = delta_p_j;
			}

			if( nd % 1000 == 0) {
				cout << "Now training.." << endl;
				std::vector<int> pred_label = predict( data);
				cout << "iter : " << iter << ", accuracy : " << calcMNIST_test_error( data, pred_label) << endl;
			}
		}
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