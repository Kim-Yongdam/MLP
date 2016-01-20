#ifndef __MLP_HPP__
#define __MLP_HPP__

#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/core/core.hpp>
#include "util.hpp"


double derivate_sigmoid( const std::vector<double>& w, const std::vector<double>& x);
double derivate_relu( const std::vector<double>& w, const std::vector<double>& x);
double derivate_softmax( const std::vector<double>& w, const std::vector<double>& x);
double sigmoid( const std::vector<double>& w, const std::vector<double>& x);
double relu( const std::vector<double>& w, const std::vector<double>& x);
double softmax( const std::vector<double>& w, const std::vector<double>& x);


class cLayer {
	private:
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

		void forward_prop( const std::vector<double>& x, std::vector<double>& output);
		void backward_prop( const std::vector<double>& x, std::vector<double>& output);


		void dataNormalization( ) {
		}
};


class cMLP {
	private:
		std::vector< cLayer> layers;

	public:

		cMLP( std::vector<int> neuron_nums) {

			// 히든 레이어 추가
			for( int nlayer = 0 ; nlayer < neuron_nums.size() - 2 ; nlayer++) {
				cLayer layer = cLayer( neuron_nums[ nlayer], neuron_nums[ nlayer + 1], sigmoid, derivate_sigmoid);
				layer.initWeightUsingNguyenWidrow();
				layers.push_back( layer);
			}

			// 아웃풋 레이어 추가
			cLayer layer = cLayer( neuron_nums[ neuron_nums.size() - 2], neuron_nums[ neuron_nums.size() - 1], softmax, derivate_softmax);
			layer.initWeightUsingNguyenWidrow();
			layers.push_back( layer);
		}


		void train( const std::vector< datum>& data, const int iter, const double learning_rate);
		std::vector<int> predict( const std::vector< datum>& data);
};


#endif