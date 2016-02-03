#ifndef __MLP_HPP__
#define __MLP_HPP__

#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/core/core.hpp>
#include "util.hpp"
#include <string.h>

void derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x, double&, double&);
void derivate_relu( const std::vector<double>& w, const std::vector<double>& x, double&, double&);
double derivate_sigmoid_one_dim( double val);
double sigmoid( const std::vector<double>& w, const std::vector<double>& x);
double relu( const std::vector<double>& w, const std::vector<double>& x);
double relu_val(double);
double derivate_relu_one_dim(double);

class cLayer {

protected:
	const int input;
	const int output;
	std::vector< std::vector<double>> w2;

public:

	cLayer( int _input, int _output) : input( _input), output( _output) {

		w2.resize( output);
		for( int noutput = 0 ; noutput < output ; noutput++)
			w2[ noutput].resize( input + 1);	// +1은 bias
	}

	void initWeightUsingUniformDistribution( double low, double high) {

		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution( low, high);
		for( int noutput = 0 ; noutput < w2.size() ; noutput++) {

			auto& w = w2[ noutput];
			for( int ninput = 0 ; ninput < w.size() ; ninput++)
				w[ ninput] = distribution( generator);
		}
	}

	void initWeightUsingNguyenWidrow( ) {

		initWeightUsingUniformDistribution( -1, 1);

		double beta = output > 2 ? 0.7 * pow( input, 1 / (double)output) : 1.;
		for( int noutput = 0 ; noutput < w2.size() ; noutput++) {

			auto& w = w2[ noutput];

			double norm = 0;
			for( int ninput = 0 ; ninput < w.size() ; ninput++)
				norm += w[ ninput] * w[ ninput];

			for( int ninput = 0 ; ninput < w.size() ; ninput++)
				w[ ninput] = beta * w[ ninput] / norm;
		}
	}

	virtual void forward_prop( const std::vector<double>& x, std::vector<double>& output) = 0;
	std::vector< std::vector<double>>& getW2( ) {
		return w2;
	}

	int getInputDim( ) {
		return input;
	}

	int getOutputDim( ) {
		return output;
	}

	/* Dropout the W2
	std::vector<std ::vector<double>>& setDropoutW2(int size) {

		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0, size);
		vector<vector <double>> w;
		for( int noutput = 0 ; noutput < w2.size() ; noutput++) {

			//	w = w2[distribution( generator) ];
		}

		return w;

	}
	*/
	void dataNormalization( ) {
	}
};



class cHiddenLayer : public cLayer{

public:
	double (*active_func)( const std::vector<double>&, const std::vector<double>&);
	void (*derivate_active_func)( const std::vector<double>&, const std::vector<double>&, double&, double&);
	double (*derivate_one_dim)( double val);
	double dropout_rate;

	cHiddenLayer( int _input, int _output,
		double (*_active_func)( const std::vector<double>&, const std::vector<double>&), double _dropout_rate = 0)
		: cLayer( _input, _output) {
			if( _active_func == sigmoid) {
				active_func = _active_func;
				derivate_active_func = derivate_sigmoid;
				derivate_one_dim = derivate_sigmoid_one_dim;
			}
			else if( _active_func == relu) {
				active_func = _active_func;
				derivate_active_func = derivate_relu;
				derivate_one_dim = derivate_relu_one_dim;
			}

			dropout_rate = _dropout_rate;
	}

	void forward_prop( const std::vector<double>& x, std::vector<double>& output);
	//vector<vector<double>> set_dropout_layer();
};


/*
class cSoftMaxLayer : public cLayer{

public:
cSoftMaxLayer( int _input, int _output,
double (*_active_func)( const std::vector<double>&, const std::vector<double>&), 
double (*_derivate_active_func)( const std::vector<double>&, const std::vector<double>&))
: cLayer( _input, _output, _active_func, _derivate_active_func) {

assert( _active_func == softmax);
assert( _derivate_active_func == derivate_softmax);
}

void forward_prop( const std::vector<double>& x, std::vector<double>& output);
};
*/

struct sLayer_shape {
	int input;
	int output;
	double dropout_rate;

	sLayer_shape( int _input, int _output, double _dropout_rate) {
		input = _input;
		output = _output;
		dropout_rate = _dropout_rate;
	}
};

class cMLP {
private:
	bool is_train;
	std::vector< cHiddenLayer*> layers;

public:

	cMLP( std::vector<int> neuron_nums) {

		is_train = false;

		// 히든 레이어 추가
		for( int nlayer = 0 ; nlayer < neuron_nums.size() - 2 ; nlayer++) {
			cHiddenLayer* layer = new cHiddenLayer( neuron_nums[ nlayer], neuron_nums[ nlayer + 1], sigmoid);

			// 1단계
			layer->initWeightUsingNguyenWidrow();
			layers.push_back( layer);
		}

		cHiddenLayer* layer = new cHiddenLayer( neuron_nums[ neuron_nums.size() - 2], neuron_nums[ neuron_nums.size() - 1], sigmoid);
		layer->initWeightUsingNguyenWidrow();
		layer->initWeightUsingNguyenWidrow();
		layers.push_back( layer);
	}

	

	cMLP( std::vector< sLayer_shape> l_shape) {

		is_train = false;

		// 히든 레이어 추가
		for( int nlayer = 0 ; nlayer < l_shape.size() ; nlayer++) {
			cHiddenLayer* layer = new cHiddenLayer( l_shape[ nlayer].input, l_shape[ nlayer].output, sigmoid, l_shape[ nlayer].dropout_rate);

			layer->initWeightUsingNguyenWidrow();
			layers.push_back( layer);
		}
	}

	/*
		Adjusting the hiddenlayer.
		Dropout the half-size.
	*/
	/*
	std::vector<cHiddenLayer*, std::allocator<cHiddenLayer *>> set_dropout_layer (int layer_size) {

		std::default_random_engine generator;
		std::uniform_real_distribution<int> distribution(0, layer_size);

		auto& hidden_layer = layers[1];
		//auto& dropout_layer;

		for( int noutput = 0 ; noutput < layer_size ; noutput++) {

			dropout_layer[noutput] = hidden_layer[ distribution( generator) ];
		}

		return dropout_layer;

	}
	*/

	void train( const std::vector< datum>& data, const int iter, const double learning_rate, const int show_train_error_interval,
		const double momentum, const double L1, const double L2);
	std::vector<int> predict( const std::vector< datum>& data);

	const std::vector< cHiddenLayer*> getLayer() {
		return layers;
	}

	enum TYPE { SIGMOID, RELU};
	void saveModel( std::string save_file_name) {

		if( is_train == false) {
			std::cout << "학습 먼저" << std::endl;
			assert( 0);
		}
		else {
			std::ofstream fout( save_file_name, std::ios::binary);

			int layers_size = layers.size();
			fout.write( reinterpret_cast<const char *> ( &layers_size), sizeof( layers_size));

			for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++) {
				const auto& w2 = layers[ nlayer]->getW2();

				int type = SIGMOID;
				fout.write( reinterpret_cast<const char *> ( &type), sizeof( type));

				int input = w2[0].size() - 1; // -1 bias 제외
				int output = w2.size();
				fout.write( reinterpret_cast<const char *> ( &input), sizeof( input));
				fout.write( reinterpret_cast<const char *> ( &output), sizeof( output));

				for( int nk = 0 ; nk < w2.size() ; nk++) {
					const auto& w = w2[ nk];
					for( int nj = 0 ; nj < w.size() ; nj++)
						fout.write( reinterpret_cast<const char *> ( &( w[ nj])), sizeof( w[ nj]));
				}
			}
		}
	}

	void loadModel( std::string load_file_name) {
		is_train = true;
		if( layers.size() > 0) {
			for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++)
				delete layers[ nlayer];
			layers.clear();
		}

		std::ifstream fin( load_file_name, std::ios::binary);

		int layer_size;
		char* str;
		fin.read( reinterpret_cast<char *> ( &layer_size), sizeof( layer_size));

		for( int nlayer = 0 ; nlayer < layer_size ; nlayer++) {

			int activation_type;
			fin.read( reinterpret_cast<char *> ( &activation_type), sizeof( activation_type));

			int input, output;
			fin.read( reinterpret_cast<char *> ( &input), sizeof( input));
			fin.read( reinterpret_cast<char *> ( &output), sizeof( output));

			double (*active_func)( const std::vector<double>&, const std::vector<double>&) = NULL;
			if( activation_type == SIGMOID)
				active_func = sigmoid;
			else if( activation_type == RELU)
				active_func = relu;
			else
				assert( 0);

			cHiddenLayer *layer = new cHiddenLayer( input, output, active_func);
			auto& w2 = layer->getW2();
			for( int nk = 0 ; nk < w2.size() ; nk++) {
				auto& w = w2[ nk];
				for( int nj = 0 ; nj < w.size() ; nj++)
					fin.read( reinterpret_cast<char *> ( &( w[ nj])), sizeof( w[ nj]));
			}

			layers.push_back( layer);
		}
	}

	~cMLP( ) {
		for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++)
			delete layers[ nlayer];
	}
};


#endif