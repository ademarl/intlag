
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


// BLAS adaptation for interval

#ifndef CUDA_BLAS_H
#define CUDA_BLAS_H


#include "cuda_blas_functions.h"
#include "cuda_interval/cuda_interval_lib.h"
#include "aux/device_data.h"


namespace intlag {




// Copies x into y
template<class T, class U> inline __host__
void acopy(int n, CudaInterval<T> const *x, CudaInterval<U> *y) {
		for(int i = 0; i < n; ++i)
			y[i] = x[i];
}


template<class T, class U> inline __host__
void acopy(int n, Interval<T> const *x, CudaInterval<U> *y) {
		for(int i = 0; i < n; ++i)
			y[i] = x[i];
}

template<class T, class U> inline __host__
void acopy(int n, CudaInterval<T> const *x, Interval<U> *y) {
		for(int i = 0; i < n; ++i)
			y[i] = x[i];
}


template<class T> inline __host__
void swap(int n, CudaInterval<T> *x, CudaInterval<T> *y) {

		for(int i = 0; i < n; ++i)
			std::swap(x[i], y[i]);
}


/*----------------------------------------- Blas Functions -------------------*/


// Using CudaGeneral for all blas kernels
typedef  CudaGeneralManaged CudaScal;
typedef  CudaGeneralManaged CudaAxpy;
typedef  CudaGeneralManaged CudaAsum;
typedef  CudaGeneralManaged CudaDot;
typedef  CudaGeneralManaged CudaNorm2;
typedef  CudaGeneralManaged CudaRot;
typedef  CudaGeneralManaged CudaRotm;

typedef  CudaGeneralManaged CudaGer;
typedef  CudaGeneralManaged CudaSyr;
typedef  CudaGeneralManaged CudaSyr2;
typedef  CudaGeneralManaged CudaSpr;
typedef  CudaGeneralManaged CudaSpr2;
typedef  CudaSharedManaged CudaGemv;
typedef  CudaGeneralManaged CudaGbmv;
typedef  CudaGeneralManaged CudaSymv;
typedef  CudaGeneralManaged CudaSbmv;
typedef  CudaGeneralManaged CudaSpmv;
typedef  CudaGeneralManaged CudaTrmv;
typedef  CudaGeneralManaged CudaTbmv;
typedef  CudaGeneralManaged CudaTpmv;
typedef  CudaGeneralManaged CudaTrsv;
typedef  CudaGeneralManaged CudaTbsv;
typedef  CudaGeneralManaged CudaTpsv;

typedef  CudaSharedManaged CudaGemm;
typedef  CudaGeneralManaged CudaSymm;
typedef  CudaGeneralManaged CudaSyrk;
typedef  CudaGeneralManaged CudaSyr2k;
typedef  CudaGeneralManaged CudaTrmm;
typedef  CudaGeneralManaged CudaTrsm;


/*----------------------------------------- Blas 1 Functions -----------------*/

//Scalar Product: x = alpha*x
template<class T, class F> inline __host__
void scal(int n, F alpha, CudaInterval<T> *x) {
  CudaScal::scal<T,F>(n, alpha, x);
}


// AXPY: alpha*x + y
template<class T, class F> inline __host__
void axpy(int n, F alpha, CudaInterval<T> const *x, CudaInterval<T> *y) {
  CudaAxpy::axpy<T,F>(n, alpha, x, y);
}


// Sum of all elements
template<class T> inline __host__
void asum(int n, CudaInterval<T> *ret, CudaInterval<T> const *x) {
  CudaAsum::asum<T>(n, ret, x);
}


// Dot Product
template<class T> inline __host__
void dot(int n, CudaInterval<T> *ret, CudaInterval<T> const *x, CudaInterval<T> const *y) {
  CudaDot::dot<T>(n, ret, x, y);
}


// Norm2
template<class T> inline __host__
void norm2(int n, CudaInterval<T> *ret, CudaInterval<T> const *x) {
  CudaNorm2::norm2<T>(n, ret, x);
}


// Rot
template<class T, class F> inline __host__
void rot(int n, CudaInterval<T>* x, CudaInterval<T>* y, F c, F s) {
  CudaRot::rot<T>(n, x, y, c, s);
}


// Rotm
template<class T, class F> inline __host__
void rotm(int n, CudaInterval<T>* x, CudaInterval<T>* y, CudaInterval<F>* H) {
  CudaRotm::rotm<T>(n, x, y, H);
}



/*----------------------------------------- Blas 2 Functions -----------------*/


// General Matrix-Vector multiplication y = alpha*A*x + beta*y
template<class T, class F> inline __host__
void ger(int n, int m, F alpha, CudaInterval<T> const *x, CudaInterval<T> const *y, CudaInterval<T> *A) {
  CudaGer::ger<T,F>(n, m, alpha, x, y, A);
}


// General Matrix-Vector multiplication y = alpha*A*x + beta*y
template<class T, class E, class F> inline __host__
void gemv(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {
  CudaGemv::gemv<T,E,F>(n, m, alpha, beta, A, x, y);
}



/*----------------------------------------- Blas 3 Functions -----------------*/


// General Matrix-Matrix multiplication C = alpha*A*B + beta*C
template<class T, class E, class F> inline __host__
void gemm(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T> *C) {
  CudaGemm::gemm<T,E,F>(n, m, p, alpha, beta, A, B, C);
}


} // namespace intlag


#endif



