#include "util.hpp"

using namespace std;
using namespace cv;


#include <io.h>
bool hasEnding( std::string const &fullString, std::string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}
std::vector<std::string> listDir( const std::string infolder, const std::string ext) {
	_finddata_t fd;
	long handle;
	int result = 1;

	std::string path = infolder;
	std::string search_path = path + "*.*";
	handle = _findfirst( search_path.c_str(), &fd);

	std::vector< std::string> vec_name;
	handle = _findfirst( search_path.c_str(), &fd);
	while( result != -1) {
		if( hasEnding ( fd.name, ext))
			vec_name.push_back( fd.name);

		result = _findnext( handle, &fd);
	}
	return vec_name;
}
//Calculate the value by multiplying 2 vectors
double vector_multiplication(const std::vector<double> &a, const std::vector<double> &b) {

	// a = w, b = x

	//cout << "Now multiplying.." << endl;

	//	for(int i=0; i<a.size(); i++)
	//	cout << a[i] << endl;

	double value=0;

	for(int i = 0; i < b.size(); i++){
		value += a[i] * b[i];
		
	}
	value += a[ b.size()] * 1;
	//	cout << value;

	return value;
}

//Calculate the index of array which contains the maximum value.

int getMaxInt(std::vector<double> &array) {

	double max = array[0];
	int index = 0;

	for (int i = 1; i < array.size(); i++) {
		if (array[i] > max) {
			max = array[i];
			index = i;
		}
	}

	return index;
}


int reverseInt (int i) 
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void getMNIST( std::string path, std::vector< datum>& train_data, const int train_data_size, std::vector< datum>& test_data, const int test_data_size) {

	train_data.clear();
	test_data.clear();

	int number_of_images = 0;
	int rows = 0;
	int cols = 0;
	unsigned int unused_val;
	unsigned char val = 0;


	ifstream fin_train_img( path + "/train-images.idx3-ubyte", ios::binary);
	ifstream fin_train_label( path + "/train-labels.idx1-ubyte", ios::binary);
	assert( fin_train_img.is_open());
	assert( fin_train_label.is_open());

	fin_train_img.read( (char*)&unused_val, sizeof( unused_val)); 
	fin_train_img.read( (char*)&number_of_images, sizeof( number_of_images));
	number_of_images = reverseInt( number_of_images);
	fin_train_img.read( (char*)&rows, sizeof( rows));
	rows = reverseInt( rows);
	fin_train_img.read( (char*)&cols, sizeof(  cols));
	cols = reverseInt( cols);

	fin_train_label.read( (char*)&unused_val, sizeof( unused_val));
	fin_train_label.read( (char*)&unused_val, sizeof( unused_val));

	for( int nimg = 0 ; nimg < train_data_size ; nimg++) {
		int idx = 0;
		datum ldatum;
		ldatum.x.resize( rows * cols);
		std::vector< double>& x = ldatum.x;
		for( int row = 0 ; row < rows ; row++) {			
			for( int col = 0 ; col < cols ; col++) {
				fin_train_img.read((char*)&val, sizeof( val));
				x[ idx++] = val;
			}
		}
		fin_train_label.read( (char*)&val, sizeof( val));
		ldatum.label = val;

		train_data.push_back( std::move( ldatum));
	}
	fin_train_img.close();
	fin_train_label.close();


	ifstream fin_test_img( path + "/t10k-images.idx3-ubyte", ios::binary);
	ifstream fin_test_label( path + "/t10k-labels.idx1-ubyte", ios::binary);
	assert( fin_test_img.is_open());
	assert( fin_test_label.is_open());

	fin_test_img.read( (char*)&unused_val, sizeof( unused_val)); 
	fin_test_img.read( (char*)&number_of_images, sizeof( number_of_images));
	number_of_images = reverseInt( number_of_images);
	fin_test_img.read( (char*)&rows, sizeof( rows));
	rows = reverseInt( rows);
	fin_test_img.read( (char*)&cols, sizeof( cols));
	cols = reverseInt( cols);

	fin_test_label.read( (char*)&unused_val, sizeof( unused_val));
	fin_test_label.read( (char*)&unused_val, sizeof( unused_val));

	for( int nimg = 0 ; nimg < test_data_size ; nimg++) {
		int idx = 0;
		datum ldatum;
		ldatum.x.resize( rows * cols);
		std::vector< double>& x = ldatum.x;
		for( int row = 0 ; row < rows ; row++) {			
			for( int col = 0 ; col < cols ; col++) {
				fin_test_img.read((char*)&val, sizeof( val));
				x[ idx++] = val;
			}
		}
		fin_test_label.read( (char*)&val, sizeof( val));
		ldatum.label = val;

		test_data.push_back( std::move( ldatum));
	}

	fin_test_img.close();
	fin_test_label.close();
}


double calcMNIST_test_error( const std::vector< datum>& data, const std::vector<int> pred_y) {

	assert( data.size() == pred_y.size());

	int match_cnt = 0;
	for( int nd = 0 ; nd < data.size() ; nd++) {
		if( data[ nd].label == pred_y[ nd])
			match_cnt++;
	}

	return match_cnt / (double)data.size();
}


void displayIMG( const std::vector< datum>& data, const int idx) {

	assert( idx < data.size());
	assert( 784 == data[ 0].x.size());

	cv::Mat src = cv::Mat( data[ idx].x);
	src = src.reshape( 0, 28);
	src.convertTo( src, CV_8UC1);
	imshow( "src", src);
	cv::waitKey();
}