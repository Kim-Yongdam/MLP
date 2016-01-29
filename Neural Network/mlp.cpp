#include "mlp.hpp"

using namespace std;
using namespace cv;


//http://www.aistudy.com/neural/MLP_kim.htm#_bookmark_165f298
//Multi-Layer Perceptron 참조 페이지

//#define LEARNING_RATE 0.3 //typical value : 0.3. 0.01과 0.001로 테스트해봤는데, 학습효과도 적고, 굉장히 느리다.
#define MOMENTUM 0.9
#define CLASS_SIZE 10
//#define ITER_COUNT 1000
#define MINI_BATCH_SIZE 17
#define LAMBDA 0.01

/*
Note : 2016-01-27

마무리 된 것들 : 

1. Momentum : control the previous weight to prevent from converging local minima or saddle point. Typically, set to 0.9 //Done(0127)
2. Mini-batch
: It controls how many times we update parameter update in how many iterations.
//왜인지는 모르겠지만, mini-batch size 17, learning_rate 0.99, iteration 100000일때 89%를 보임. 다른 hyperparameter값으로는 도달하지 못함.
+ 왜 90%를 못넘어가는지 모르겠음.(Almost done 0128)

추가해야 할 사항 :

1. Regularization terms

1) L1, L2 : filtering term. It also prevent from converging local minima or saddle point.



3. Drop-out
: It adjusts the speed of learning by deleting some edges(weights) in learning.(randomly choosing in each learning)

4. Data-Normalization
: When we use the activation function such as, sigmoid and ReLU, we should put together the data sets to manage to learning.

*/


void derivate_relu( const std::vector<double>& w, const std::vector<double>& x, double& activation, double& derivation) {

	double z = vector_multiplication(w, x);

	activation = max(0., z);

	if(z >= 0.) derivation = 1;
	else derivation = 0;	
}

double relu( const std::vector<double>& w, const std::vector<double>& x) {

	return max(0., vector_multiplication(w,x));
}

double derivate_relu_one_dim(double val) {

	if(val >= 0) return 1;
	else return 0;

}

void derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x, double& activation, double& derivation) {

	activation = 1 / (1 + exp( -vector_multiplication(w, x)));
	derivation = activation * ( 1 - activation);
}


double sigmoid( const std::vector<double>& w, const std::vector<double>& x) {

	return 1 / (1 + exp( -vector_multiplication(w, x)));
}


double sigmoid_val( double val) {
	return 1 / ( 1 + exp( -val));
}


double derivate_sigmoid_one_dim( double val) {
	return sigmoid_val( val) * ( 1 - sigmoid_val( val));
}


/*
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
*/


void cHiddenLayer::forward_prop( const std::vector<double>& x, std::vector<double>& output) {

	output.resize( w2.size());
	for(int i =0; i < w2.size(); i++)
		output[i] = active_func(w2[i], x);
}


std::vector< datum> getMINIBATCH( const std::vector< datum>& data, const int minibatch_size) {

	static int idx = 0;

	std::vector< datum> mini_batch;
	for( int nd = 0 ; nd < minibatch_size ; nd++) {

		idx %= data.size();
		mini_batch.push_back( data[ idx]);
		idx++;

	}

	return mini_batch;
}


// back-prop
void cMLP::train( const std::vector< datum>& data, const int iteration, const double learning_rate, const int show_train_error_interval,
				 const int L1, const int L2) {

	for( int iter = 0 ; iter < iteration ; iter++) {

		auto& mini_batch_data = getMINIBATCH( data, MINI_BATCH_SIZE);
		vector< vector<double>> save_o_pk( layers.size() + 1); // 입력층 + ( 히든 + 출력)레이어 해서 layers.size() + 1	
		vector< vector<double>> save_f_prime_pk( layers.size() + 1); // 입력층 + ( 히든 + 출력)레이어 해서 layers.size() + 1
		vector<double> delta_p_k1;

		save_o_pk[ 0].resize( data[0].x.size() + 1, 0);
		save_f_prime_pk[ 0].resize( save_o_pk[ 0].size(), 0);

		for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++) {
			auto& layer = layers[ nlayer];
			auto& w2 = layer->getW2();

			save_f_prime_pk[ nlayer + 1].resize( w2.size() + 1, 0);
			save_o_pk[ nlayer + 1].resize( w2.size() + 1, 0);
		}

		delta_p_k1.resize( layers[ layers.size() -1]->getW2().size(), 0);


		for(int batch_iter = 0; batch_iter < MINI_BATCH_SIZE; batch_iter++) {
			const auto& d = mini_batch_data[ batch_iter];

			vector<double> output1 = d.x;

			for( int noutput = 0 ; noutput < output1.size() ; noutput++)
				save_o_pk[ 0][ noutput] += output1[ noutput];
			save_o_pk[ 0][ output1.size()] += 1;

			for( int ninput = 0 ; ninput < save_o_pk[ 0].size() ; ninput++)
				save_f_prime_pk[ 0][ ninput] += layers[ 0]->derivate_one_dim( save_o_pk[ 0][ ninput]);
			for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++) {
				auto& layer = layers[ nlayer];

				vector<double> output2;
				const auto& w2 = layer->getW2();
				output2.resize( w2.size());
				for(int i =0; i < w2.size(); i++) {
					//output2[i] = layer->active_func( w2[i], output1);

					double derivation;
					layer->derivate_active_func( w2[i], output1, output2[i], derivation);
					save_f_prime_pk[ nlayer + 1][ i] += derivation;
				}
				save_f_prime_pk[nlayer + 1][ w2.size()] += layer->derivate_one_dim( 1);

				output1 = output2;
				for( int noutput = 0 ; noutput < output1.size() ; noutput++)
					save_o_pk[ nlayer + 1][ noutput] += output1[ noutput];
				save_o_pk[ nlayer + 1][ output1.size()] += 1;
			}


			// 5단계 -------------------------------------------------------------------------------------
			//one hot encoding
			vector<double> t_p1( output1.size(), 0);
			t_p1[ d.label] = 1;			// 실제출력 값

			//	vector<double> delta_p_k1( output1.size());
			for( int ndelta = 0 ; ndelta < delta_p_k1.size() ; ndelta++){
				//L1, L2 regularization
				delta_p_k1[ ndelta] += (( t_p1[ ndelta] - output1[ ndelta]) * save_f_prime_pk[ save_f_prime_pk.size() - 1][ ndelta]);
			}
			// 5단계 끝====================================================================================

			for( int n = 0 ; n < save_f_prime_pk[ layers.size()].size() ; n++)
				save_f_prime_pk[ layers.size()][ n] = 0;
		}//mini-batch iteration

		std::for_each( delta_p_k1.begin(), delta_p_k1.end(), [] ( double& val) { val /= MINI_BATCH_SIZE;});
		for( int nlayer = 0 ; nlayer < layers.size() + 1 ; nlayer++) {

			auto& save_o_pk_element = save_o_pk[ nlayer];
			std::for_each( save_o_pk_element.begin(), save_o_pk_element.end(), [] ( double& val) { val /= MINI_BATCH_SIZE;});
			auto& save_f_prime_pk_element = save_f_prime_pk[ nlayer];
			std::for_each( save_f_prime_pk_element.begin(), save_f_prime_pk_element.end(), [] ( double& val) { val /= MINI_BATCH_SIZE;});
		}


		// 6단계 ~ 8단계
		for( int nlayer = layers.size() - 1 ; nlayer >= 0 ; nlayer--) {

			auto& layer = layers[ nlayer];
			auto& w2 = layer->getW2();
			vector<double> delta_p_j(257);
			int current_layer_size = save_o_pk[nlayer].size();
			int previous_layer_size = save_o_pk[nlayer+1].size();
			double weight_change;
			double L1_term;
			double L2_term;
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
					if(nlayer == 1) delta_p_j[j] += delta_p_k1[k] * w2[k][j] * save_f_prime_pk[ nlayer][ j];
					L1_term = L1 * LAMBDA * (w2[k][j] / abs(w2[k][j])) / current_layer_size;
					L2_term = L2 * LAMBDA * w2[k][j] / current_layer_size;
					weight_change = learning_rate * (delta_p_k1[k] + L1_term + L2_term) * save_o_pk[nlayer][j];
					w2[k][j] = w2[k][j] + (MOMENTUM * weight_change);

				}
			}
			// delta_p_j 구해서 이전 레이어로 전파
			delta_p_k1 = delta_p_j;
		}

		if( iter % show_train_error_interval == 0) {
			cout << "Now training.." << endl;
			std::vector<int> pred_label = predict( data);
			cout << "iter : " << iter << ", accuracy : " << calcMNIST_test_error( data, pred_label) << endl;
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