//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef TEST_CASES_OMP_H
#define TEST_CASES_OMP_H

#include <omp.h>
#include <boost/numeric/interval.hpp>
using boost::numeric::interval;
using namespace boost::numeric;


void omp_scal(double alpha, interval<double>* x, int n) {
	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		x[j] *= alpha;
}

void omp_sum(interval<double>* x, interval<double>* y, int n) {
	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		y[j] += x[j];
}

void omp_mult(interval<double>* x, interval<double>* y, int n) {
	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		y[j] *= x[j];
}

void omp_axpy(double alpha, interval<double>* x, interval<double>* y, int n) {
	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		y[j] += alpha*x[j];
}


// HACK: OpenMP implementation uses indirect reduction since intervals are not considered scalar by OpenMP, maybe we can do better
void omp_asum(interval<double>* x, int n) {
	double r1 = 0;
	double r2 = 0;
	#pragma omp parallel for reduction(+:r1, r2)
	for (int j = 0; j < n; ++j){
		interval<double> r(r1, r2);
		r += x[j];
		r1 = lower(r);
		r2 = upper(r);
	}
}


void omp_dot(interval<double>* x, interval<double>* y, int n) {
	double r1 = 0;
	double r2 = 0;
	#pragma omp parallel for reduction(+:r1, r2)
	for (int j = 0; j < n; ++j){
		interval<double> r(r1, r2);
		r += abs(x[j])*abs(y[j]);
		r1 = lower(r);
		r2 = upper(r);
	}
	interval<double> r(r1, r2);
}


void omp_norm2(interval<double>* x, interval<double>* y, int n) {
	double r1 = 0;
	double r2 = 0;
	#pragma omp parallel for reduction(+:r1, r2)
	for (int j = 0; j < n; ++j) {
		interval<double> r(r1, r2);
		r += abs(x[j])*abs(x[j]);
		r1 = lower(r);
		r2 = upper(r);
	}
	interval<double> r(r1, r2);
	r = sqrt(r);
}

void omp_gemv(int n, int m, double alpha, double beta, interval<double> *A, interval<double> *x, interval<double> *y) {

	#pragma omp parallel for
	for (int j = 0; j < m; ++j)
		x[j] = alpha * x[j];

	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		interval<double> sum = 0;
		for (int j = 0; j < m; ++j)
			sum += A[i*m + j]*x[j];
		y[i] = beta*y[i] + sum;
	}
}

void omp_gemm(int n, int m, int p, double alpha, double beta, interval<double> *A, interval<double> *B, interval<double> *C) {

	if (n*m < m*p) {
		#pragma omp parallel for
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				A[i*m+j] = alpha * A[i*m+j];
	}
	else {
		#pragma omp parallel for
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < p; ++j)
				B[i*p+j] = alpha * B[i*p+j];
	}

	#pragma omp parallel for
	for (int i = 0; i < n; ++i) 
		for (int j = 0; j < p; ++j) {
			interval<double> sum = 0;
			for (int k = 0; k < m; ++k)
				sum += A[i*m+k]*B[j*p+k];
			C[i*p+j] = beta*C[i*p+j] + sum;
		}
}

#endif
