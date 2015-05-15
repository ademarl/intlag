//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef BLAS_INTERVAL_RUNTIME_H
#define BLAS_INTERVAL_RUNTIME_H

#include <cstdio>
#include <cstdlib>
using namespace std;

#include <omp.h>
#include <boost/numeric/interval.hpp>
using boost::numeric::interval;
using namespace boost::numeric;

#include "interval_blas.h"
//#include "cuda_box_lib.h"
//#include "cuda_interval_lib.h"

#include "test_cases/test_cases.h"
#include "test_cases/test_cases_serial.h"
#include "test_cases/test_cases_omp.h"


#include "auxiliar/io_aux.h"
#include "auxiliar/timer_aux.h"

// NOTE: Test cases ignore the return value of expressions by not assigning the results

// TODO: Refactor input reading
// TODO: Refactor test cases of cuda cases into a function that receives a fpointer (needs a wrapper for box_gpu_lib.h)


void reset_vec_gpu(interval_gpu<double>* dx, interval_gpu<double>* x, int n) {

	for(int i = 0; i < n; ++i)
		x[i] = dx[i];
}


// Serial runtime tests
void serial_blas_runtime(int n_iter) {

	cout << "Using serial implementation..." << endl;

	int n;
	double alpha;
	interval<double> *x = NULL, *y = NULL;

	// Reads input
	cin >> alpha;
	cin >> n;
	x = new interval<double>[n];
	y = new interval<double>[n];
	read_interval_input(x, y, n);

	Timer t;

	// Test cases
	t = test_case(n_iter, n, alpha, x, &serial_scal);
	cout << "Time for the execution of " << n << " interval scalar multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, alpha, x, y, &serial_axpy);
	cout << "Time for the execution of " << n << " interval axpy's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, &serial_asum);
	cout << "Time for the execution of " << n << " interval asum's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &serial_dot);
	cout << "Time for the execution of " << n << " interval dot products " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &serial_norm2);
	cout << "Time for the execution of " << n << " interval norm2 " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	// gemm - modifies y
	t = test_case(n_iter, 1000, 1000, 2.0, 3.0, x, x, y, &serial_gemv);
	cout << "Time for the execution of " << 1000 << " x " << 1000 << " interval gemv " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	// gemm - modifies y
	t = test_case(1, 1000, 1000, 1000, 2.0, 3.0, x, x, y, &serial_gemm);
	cout << "Time for the execution of " << 1000 << " x " << 1000 << " interval gemm " << 1 << " times: " << t.getSeconds() << "s" << endl;

	delete[] x;
	delete[] y;
}


// OpenMP runtime tests
void openmp_blas_runtime(int n_iter) {

	omp_set_num_threads(8);
	cout << "Using OpenMP implementation..." << endl;

	int n;
	double alpha;
	interval<double> *x = NULL, *y = NULL;

	// Reads input
	cin >> alpha;
	cin >> n;
	x = new interval<double>[n];
	y = new interval<double>[n];
	read_interval_input(x, y, n);

	Timer t;

	// Test cases
	t = test_case(n_iter, n, alpha, x, &omp_scal);
	cout << "Time for the execution of " << n << " interval scalar multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, alpha, x, y, &omp_axpy);
	cout << "Time for the execution of " << n << " interval axpy's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, &omp_asum);
	cout << "Time for the execution of " << n << " interval asum's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &omp_dot);
	cout << "Time for the execution of " << n << " interval dot products " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &omp_norm2);
	cout << "Time for the execution of " << n << " interval norm2 " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	// gemv - modifies y
	t = test_case(n_iter, 1000, 1000, 2.0, 3.0, x, x, y, &omp_gemv);
	cout << "Time for the execution of " << 1000 << " x " << 1000 << " interval gemv " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	// gemm - modifies y
	t = test_case(1, 1000, 1000, 1000, 2.0, 3.0, x, x, y, &omp_gemm);
	cout << "Time for the execution of " << 1000 << " x " << 1000 << " interval gemm " << 1 << " times: " << t.getSeconds() << "s" << endl;

	delete[] x;
	delete[] y;

}


// CUDA runtime tests
void cuda_blas_runtime(int n_iter) {

	cout << "Using CUDA implementation..." << endl;

	int n;
	double alpha;
	intlag::interval_gpu<double> *dx = NULL, *dy = NULL, *x = NULL, *y = NULL, ret;

	// Reads input
	cin >> alpha;
	cin >> n;
	dx = new interval_gpu<double>[n];
	dy = new interval_gpu<double>[n];
	x = new interval_gpu<double>[n];
	y = new interval_gpu<double>[n];
	read_interval_input(dx, dy, n);
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	// Start up GPU
	scal(n, alpha, x);
	reset_vec_gpu(dx, x, n);


	Timer t;	
	// Test cases

	t = test_case(n_iter, n, alpha, x, &intlag::scal);
	cout << "Time for the execution of " << n << " interval scalar multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	t = test_case(n_iter, n, alpha, x, y, &intlag::axpy);
	cout << "Time for the execution of " << n << " interval axpy's " << n_iter << " times: " << t.getSeconds() << "s" << endl;
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	t = test_case(n_iter, &ret, n, x, &intlag::asum);
	cout << "Time for the execution of " << n << " interval asum's " << n_iter << " times: " << t.getSeconds() << "s" << endl;
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	t = test_case(n_iter, &ret, n, x, y, &intlag::dot);
	cout << "Time for the execution of " << n << " interval dot products " << n_iter << " times: " << t.getSeconds() << "s" << endl;
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	t = test_case(n_iter, &ret, n, x, &intlag::norm2);
	cout << "Time for the execution of " << n << " interval norm2's " << n_iter << " times: " << t.getSeconds() << "s" << endl;
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	// gemv - modifies y
	t = test_case(n_iter, 1000, 1000, 2.0, 3.0, x, x, y, &intlag::gemv);
	cout << "Time for the execution of " << 1000 << " x " << 1000 << " interval gemv " << n_iter << " times: " << t.getSeconds() << "s" << endl;
	reset_vec_gpu(dx, x, n);
	reset_vec_gpu(dy, y, n);

	// gemm - modifies y
	t = test_case(1, 1000, 1000, 1000, 2.0, 3.0, x, x, y, &intlag::gemm);
	cout << "Time for the execution of " << 1000 << " x " << 1000 << " interval gemm " << 1 << " times: " << t.getSeconds() << "s" << endl;

	delete[] dy;
	delete[] dx;
	delete[] x;
	delete[] y;
}

#endif
