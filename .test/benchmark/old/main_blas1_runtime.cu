

// This program intends to run performance tests box_gpu's BLAS 1
// input must contain an alpha, followed by the number of intervals and the intervals themselves (lower first)

#include <iostream>

#include "io_aux.h"
#include "blas1_runtime.h"


// NOTE: Double precision only
int main(int argc, char** argv) {

	std::cout << "Runtime test cases for interval BLAS level 1" << endl;

	int method = 0;
	int n_iter = 100;
	set_options(argc, argv, &method, &n_iter);

	switch(method) {
		case 0:
				serial_blas1_runtime(n_iter);
			break;
		case 1:
				openmp_blas1_runtime(n_iter);
			break;
		case 2:
				cuda_blas1_runtime(n_iter);
			break;
		
		default:
			std::cerr << "Error: Method undefined, 0-2 expected."<< endl;
			exit(1);
	}

	return 0;
}
