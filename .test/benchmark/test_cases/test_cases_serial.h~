//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef TEST_CASES_SERIAL_H
#define TEST_CASES_SERIAL_H

#include <vector>

#include <boost/numeric/interval.hpp>
using boost::numeric::interval;
using namespace boost::numeric;

void serial_scal(double alpha, interval<double>* x, int n) {
	for (int j = 0; j < n; ++j)
		x[j] = alpha * x[j];
}

void serial_sum(interval<double>* x, interval<double>* y, int n) {
	for (int j = 0; j < n; ++j)
		y[j] += x[j];
}

void serial_mult(interval<double>* x, interval<double>* y, int n) {
	for (int j = 0; j < n; ++j)
		y[j] *= x[j];
}

void serial_axpy(double alpha, interval<double>* x, interval<double>* y, int n) {
	for (int j = 0; j < n; ++j)
		y[j] += alpha*x[j];
}

void serial_asum(interval<double>* x, int n) {
	interval<double> r = 0;
	for (int j = 0; j < n; ++j)
		r += x[j];
}

void serial_dot(interval<double>* x, interval<double>* y, int n) {
	interval<double> r = 0;
	for (int j = 0; j < n; ++j)
		r += abs(x[j]) * abs(y[j]);
}

void serial_norm2(interval<double>* x, interval<double>* y, int n) {
	interval<double> r = 0;
	for (int j = 0; j < n; ++j)
		r += abs(x[j]) * abs(x[j]);

	r = sqrt(r);
}

void serial_gemv(int n, int m, double alpha, double beta, interval<double> *A, interval<double> *x, interval<double> *y) {

	for (int j = 0; j < m; ++j)
		x[j] = alpha * x[j];

	for (int i = 0; i < n; ++i) {
		interval<double> sum = 0;
		for (int j = 0; j < m; ++j)
			sum += A[i*m + j]*x[j];
		y[i] = beta*y[i] + sum;
	}
}


void serial_gemm(int n, int m, int p, double alpha, double beta, interval<double> *A, interval<double> *B, interval<double> *C) {

	if (n*m < m*p) {
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				A[i*m+j] = alpha * A[i*m+j];
	}
	else {
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < p; ++j)
				B[i*p+j] = alpha * B[i*p+j];
	}

	for (int i = 0; i < n; ++i) 
		for (int j = 0; j < p; ++j) {
			interval<double> sum = 0;
			for (int k = 0; k < m; ++k)
				sum += A[i*m+k]*B[j*p+k];
			C[i*p+j] = beta*C[i*p+j] + sum;
		}
}

#endif
