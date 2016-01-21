#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct datum {
	std::vector<double> x;
	int label;

	datum() {
	}
};

int getMaxInt(const std::vector<double> &array);
double vector_multiplication(const std::vector<double> &a, const std::vector<double> &b);
void getMNIST( std::string path, std::vector< datum>& train_data, const int train_data_size, std::vector< datum>& test_data, const int test_data_size);
double calcMNIST_test_error( const std::vector< datum>& data, const std::vector<int> pred_y);
void displayIMG( const std::vector< datum>& data, const int idx);

#endif