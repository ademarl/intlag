//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef TEST_CASES_H
#define TEST_CASES_H


#include "../auxiliar/timer_aux.h"

using namespace intlag;

/////////////////////////////////////////////// BOOST-BASED ////////////////////

typedef void (*blas_func1_t)(interval<double>*, int);
Timer test_case(int n_iter, int n, interval<double>* x, blas_func1_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(x, n);
	t.stop();
	return t;
}


typedef void (*blas_func2_t)(interval<double>*, interval<double>*, int);
Timer test_case(int n_iter, int n, interval<double>* x, interval<double>* y, blas_func2_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(x, y, n);
	t.stop();
	return t;
}


typedef void (*blas_func3_t)(double, interval<double>*, int);
Timer test_case(int n_iter, int n, double alpha, interval<double>* x, blas_func3_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(alpha, x, n);
	t.stop();
	return t;
}


typedef void (*blas_func4_t)(double, interval<double>*, interval<double>*, int);
Timer test_case(int n_iter, int n, double alpha, interval<double>* x, interval<double>* y, blas_func4_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(alpha, x, y, n);
	t.stop();
	return t;
}

typedef void (*blas_func_gemv_t)(int n, int m, double alpha, double beta, interval<double> *A, interval<double> *x, interval<double> *y);
Timer test_case(int n_iter, int n, int m, double alpha, double beta, interval<double> *A, interval<double> *x, interval<double> *y, blas_func_gemv_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(n, m, alpha, beta, A, x, y);
	t.stop();
	return t;
}


typedef void (*blas_func_gemm_t)(int n, int m, int p, double alpha, double beta, interval<double> *A, interval<double> *B, interval<double> *C);
Timer test_case(int n_iter, int n, int m, int p, double alpha, double beta, interval<double> *A, interval<double> *B, interval<double> *C, blas_func_gemm_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(n, m, p, alpha, beta, A, B, C);
	t.stop();
	return t;
}

/////////////////////////////////////////////// ~BOOST-BASED ///////////////////



/////////////////////////////////////////////// GPU-BASED //////////////////////

typedef bool (*blas_gpu1_t)(int, double, interval_gpu<double>*);
Timer test_case(int n_iter, int n, double alpha, interval_gpu<double>* x, blas_gpu1_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(n, alpha, x);
	t.stop();
	return t;
}


typedef bool (*blas_gpu2_t)(int, double, interval_gpu<double> const*, interval_gpu<double> *);
Timer test_case(int n_iter, int n, double alpha, interval_gpu<double> *x, interval_gpu<double> *y, blas_gpu2_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(n, alpha, x, y);
	t.stop();
	return t;
}


typedef bool (*blas_gpu3_t)(interval_gpu<double> *, int, interval_gpu<double> const *);
Timer test_case(int n_iter, interval_gpu<double> *ret, int n, interval_gpu<double>* x, blas_gpu3_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(ret, n, x);
	t.stop();
	return t;
}


typedef bool (*blas_gpu4_t)(interval_gpu<double> *, int, interval_gpu<double> const *, interval_gpu<double> const *);
Timer test_case(int n_iter, interval_gpu<double> *ret, int n, interval_gpu<double>* x, interval_gpu<double>* y, blas_gpu4_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i){
		f(ret, n, x, y);
	}
	t.stop();
	return t;
}


typedef bool (*blas_gemv_t)(int n, int m, double alpha, double beta, interval_gpu<double> const *A, interval_gpu<double> const *x, interval_gpu<double> *y);
Timer test_case(int n_iter, int n, int m, double alpha, double beta, interval_gpu<double> *A, interval_gpu<double> *x, interval_gpu<double> *y, blas_gemv_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(n, m, alpha, beta, A, x, y);
	t.stop();
	return t;
}


typedef bool (*blas_gemm_t)(int n, int m, int p, double alpha, double beta, interval_gpu<double> const *A, interval_gpu<double> const *B, interval_gpu<double> *C);
Timer test_case(int n_iter, int n, int m, int p, double alpha, double beta, interval_gpu<double> *A, interval_gpu<double> *B, interval_gpu<double> *C, blas_gemm_t f) {

	Timer t;
	for (int i = 0; i < n_iter; ++i)
		f(n, m, p, alpha, beta, A, B, C);
	t.stop();
	return t;
}
/////////////////////////////////////////////// ~GPU-BASED /////////////////////


#endif
