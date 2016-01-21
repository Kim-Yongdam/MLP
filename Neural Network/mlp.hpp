#ifndef __MLP_HPP__
#define __MLP_HPP__

#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/core/core.hpp>
#include "util.hpp"
#include <string.h>

double derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x);
double derivate_relu( const std::vector<double>& w, const std::vector<double>& x);
double derivate_softmax( const std::vector<double>& w, const std::vector<double>& x);
double sigmoid( const std::vector<double>& w, const std::vector<double>& x);
double relu( const std::vector<double>& w, const std::vector<double>& x);
double softmax( const std::vector<double>& w, const std::vector<double>& x);
void init();

extern std::vector<double> activation_value;		//Opk
extern std::vector<double> derivation_value;		//delta pj
extern std::vector<double> activated_output;		//Opj
extern std::vector<double> derivated_output;		//delta pk


class cLayer {

protected:
	const int input;
	const int output;
	std::vector< std::vector<double>> w2;
	double (*active_func)( const std::vector<double>&, const std::vector<double>&);
	double (*derivate_active_func)( const std::vector<double>&, const std::vector<double>&);

public:

	cLayer( int _input, int _output,
		double (*_active_func)( const std::vector<double>&, const std::vector<double>&), 
		double (*_derivate_active_func)( const std::vector<double>&, const std::vector<double>&)) : input( _input), output( _output) {

			active_func = _active_func;
			derivate_active_func = _derivate_active_func;
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
	virtual void backward_prop( const std::vector<double>& x, const std::vector<double>& sigma_p, std::vector<double>& output) = 0;

	void dataNormalization( ) {

	}
};



class cHiddenLayer : public cLayer{

public:
	cHiddenLayer( int _input, int _output,
		double (*_active_func)( const std::vector<double>&, const std::vector<double>&), 
		double (*_derivate_active_func)( const std::vector<double>&, const std::vector<double>&))
		: cLayer( _input, _output, _active_func, _derivate_active_func) {
			assert( _active_func != softmax);
			assert( _derivate_active_func != derivate_softmax);
	}

	void forward_prop( const std::vector<double>& x, std::vector<double>& output);
	void backward_prop( const std::vector<double>& x, const std::vector<double>& t_p, std::vector<double>& output);
};


// softmax만 지원!!
class cOutputLayer : public cLayer{

public:
	cOutputLayer( int _input, int _output,
		double (*_active_func)( const std::vector<double>&, const std::vector<double>&), 
		double (*_derivate_active_func)( const std::vector<double>&, const std::vector<double>&))
		: cLayer( _input, _output, _active_func, _derivate_active_func) {

			assert( _active_func == softmax);
			assert( _derivate_active_func == derivate_softmax);
	}

	void forward_prop( const std::vector<double>& x, std::vector<double>& output);
	void backward_prop( const std::vector<double>& x, const std::vector<double>& t_p, std::vector<double>& output);
};



class cMLP {
private:
	std::vector< cLayer*> layers;

public:

	cMLP( std::vector<int> neuron_nums) {

		// 히든 레이어 추가
		for( int nlayer = 0 ; nlayer < neuron_nums.size() - 2 ; nlayer++) {
			cHiddenLayer* layer = new cHiddenLayer( neuron_nums[ nlayer], neuron_nums[ nlayer + 1], sigmoid, derivate_sigmoid);

			// 1단계
			layer->initWeightUsingNguyenWidrow();
			layers.push_back( layer);
		}

		// 아웃풋 레이어 추가
		cOutputLayer* layer = new cOutputLayer( neuron_nums[ neuron_nums.size() - 2], neuron_nums[ neuron_nums.size() - 1], softmax, derivate_softmax);
		layer->initWeightUsingNguyenWidrow();
		layer->initWeightUsingNguyenWidrow();
		layers.push_back( layer);
	}


	void train( const std::vector< datum>& data, const int iter, const double learning_rate);
	std::vector<int> predict( const std::vector< datum>& data);

	~cMLP( ) {
		for( int nlayer = 0 ; nlayer < layers.size() ; nlayer++)
			delete layers[ nlayer];
	}
};


#endif