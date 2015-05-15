//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

// This program intends to run performance tests box_gpu's BLAS 1
// input must contain an alpha, followed by the number of intervals and the intervals themselves (lower first)

#include <iostream>

#include "auxiliar/io_aux.h"
#include "blas_interval_runtime.h"


// NOTE: Double precision only
int main(int argc, char** argv) {

	std::cout << "Runtime test cases for interval BLAS" << endl;

	int method = 0;
	int n_iter = 100;
	set_options(argc, argv, &method, &n_iter);

	switch(method) {
		case 0:
			serial_blas_runtime(n_iter);
			break;
		case 1:
			openmp_blas_runtime(n_iter);
			break;
		case 2:
			cuda_blas_runtime(n_iter);
			break;

		default:
			std::cerr << "Error: Method undefined, 0-2 expected."<< endl;
			exit(1);
	}

	return 0;
}
