
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BLAS_KERNELS_H
#define CUDA_BLAS_KERNELS_H

#include "cuda_interval/cuda_interval_lib.h"
#include "aux/cuda_error.h"
#include "aux/cuda_grid.h"

namespace intlag {


// method to get the device id of individual threads
inline __device__
int get_thread_id() {
	return blockIdx.x*blockDim.x + threadIdx.x;
}

inline __device__
int get_thread_idx() {
	return blockIdx.x*blockDim.x + threadIdx.x;
}

inline __device__
int get_thread_idy() {
	return blockIdx.y*blockDim.y + threadIdx.y;
}

// method to get the device id of individual threads by offset
inline __device__
int get_thread_id(int k) {
	return k*(blockIdx.x*blockDim.x + threadIdx.x);
}


//------------------------------------------------------------------------------

template <class T>
__global__ void PRINT(CudaInterval<T> const *x, int N) {
	int i = get_thread_id();
	if (i < N)
    printf("---------%f %f---------\n", x[i].inf(), x[i].sup());
}


// Kernel for copying
template <class T>
__global__ void COPY(CudaInterval<T> const *x, CudaInterval<T> *y, int N) {
	int i = get_thread_id();
	if (i < N)
		y[i] = x[i];
}

// Kernel for copying
template <class T>
__global__ void EQ(CudaInterval<T> *x, CudaInterval<T> *y) {
  x = y;
}


// Kernel for scalar multiplication
template <class T, class F>
__global__ void SCALAR_MULT(F alpha, CudaInterval<T> *x, int N) {
	int i = get_thread_id();
	if (i < N)
		x[i] = alpha*x[i];
}


// Kernel for elementwise sum
template <class T>
__global__ void ADD(CudaInterval<T> const *a, CudaInterval<T> *b, int N) {
	int i = get_thread_id();
	if (i < N)
		b[i] = a[i] + b[i];
}


// Kernel for elementwise multiplication
template <class T>
__global__ void MULT(CudaInterval<T> const *x, CudaInterval<T> *y, int N) {
	int i = get_thread_id();
	if (i < N)
		y[i] = x[i]*y[i];
}


// Kernel for elementwise fma
template <class T, class F>
__global__ void AXPY(F alpha, CudaInterval<T> const *x, CudaInterval<T> *y, int N) {
	int i = get_thread_id();
	if (i < N)
		y[i] = alpha*x[i] + y[i];
}


template <class T>
__global__ void NORMMULT(CudaInterval<T> const *x, CudaInterval<T> *y, int N) {
	int i = get_thread_id();
	if (i < N)
		y[i] = abs(x[i]) * abs(y[i]);
}


template <class T>
__global__ void ASUM(CudaInterval<T> const *x, CudaInterval<T> *y, int N) {
	int i = get_thread_id(2);
	if (i < N)
		y[i/2] = abs(x[i]);
	if (i+1 < N)
		y[i/2] = y[i/2] + abs(x[i+1]);
}

// SQRT of one element, if the lower bound is negative, square root is not defined
template <class T>
__global__ void SQRT(CudaInterval<T> const *x, CudaInterval<T>* y) {
	if (x->inf() < 0) { y = NULL; return; }
  *y = sqrt(*x);
}


// ROT
template <class T, class F>
__global__ void ROT(CudaInterval<T>* x, CudaInterval<T>* y, F c, F s, int N) {
	int i = get_thread_id();
	if (i < N) {
    CudaInterval<T> aux  = c*x[i] + s*y[i];
    y[i] = c*y[i] - s*x[i];
    x[i] = aux;
  }
}


// ROTM case h00 = 1.0
template <class T, class F>
__global__ void ROTM1(CudaInterval<T>* x, CudaInterval<T>* y, F const h1, F const h4, int N) {
	int i = get_thread_id();
	if (i < N) {
    CudaInterval<T> aux = h1*x[i] + y[i];
    y[i] = h4*y[i] - x[i];
    x[i] = aux;
  }
}


// ROTM case h00 = 0.0
template <class T, class F>
__global__ void ROTM0(CudaInterval<T>* x, CudaInterval<T>* y, F const h2 , F const h3, int N) {
	int i = get_thread_id();
	if (i < N) {
    CudaInterval<T> aux = x[i] + h3*y[i];
    y[i] = h2*x[i] + y[i];
    x[i] = aux;
  }
}


// ROTM case h00 = -1.0
template <class T, class F>
__global__ void ROTMM1(CudaInterval<T>* x, CudaInterval<T>* y, F const h1, F const h2 , F const h3 , F const h4, int N) {
	int i = get_thread_id();
	if (i < N) {
    CudaInterval<T> aux = h1*x[i] + h3*y[i];
    y[i] = h2*x[i] + h4*y[i];
    x[i] = aux;
  }
}

// GER, returns in A
template <class T, class F>
__global__ void GER(int n, int m, F alpha, CudaInterval<T> const *x, CudaInterval<T> const *y, CudaInterval<T>* A) {
	int row = get_thread_idx();
	int col = get_thread_idy();
	if (row < n && col < m) {
    A[row*m+col] = A[row*m+col] + alpha*x[row]*y[col];
  }
}


// GEMV, returns in y
template <class T, class E, class F>
__global__ void GEMV(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {
	int i = get_thread_id();
	if (i < n) {

		CudaInterval<T> sum(0);
		for(int k = 0; k < m; ++k)
			sum = sum + A[i*m+k] * x[k];
		y[i] = alpha*sum + y[i];
	}
}


// GEMV, returns in y
template <class T, class E, class F>
__global__ void GEMV_SHARED(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {

  int i = get_thread_id();
  int it = threadIdx.x;
  __shared__ CudaInterval<T> xs[BLOCK_SIZE];

  CudaInterval<T> sum = 0.0;
  __syncthreads();

  int NUM_BLOCKS = (m - 1)/BLOCK_SIZE+1;

  for (int j = 0; j < NUM_BLOCKS; ++j) {
    int k = j * BLOCK_SIZE + it;
    if ((k) <  m) xs[it] = x[k];
    else xs[it] = 0.0;

    __syncthreads();

    int offset = i*m + k-it;
    for (int l = 0; l < BLOCK_SIZE; ++l)
      sum = sum + A[offset+l] * xs[l]; //Row-major
      //sum = sum + A[i + (l + BLOCK_SIZE * j) * n] * xs[l]; // Column-major

    __syncthreads();
  }

  if (i < n) y[i] = alpha*sum + y[i];
}


// GEMM, returns in C
template <class T, class E, class F>
__global__ void GEMM(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T>* C) {
	int row = get_thread_idx();
	int col = get_thread_idy();
	if (row < n && col < m) {

    int cindex = row*m + col;
    row = row*p;
		CudaInterval<T> sum(0.0);
		for(int k = 0; k < p; ++k)
			sum = sum + A[row + k] * B[col + k*m];

		C[cindex] = alpha*sum + C[cindex];
	}
}


// SYMM, returns in C
template <class T, class E, class F>
__global__ void SYMM(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T>* C) {
	int row = get_thread_idx();
	int col = get_thread_idy();
	if (row < n && col < m) {

    int cindex = row*m + col;
    row = row*p;
		CudaInterval<T> sum(0.0);
		for(int k = 0; k < p; ++k)
			sum = sum + A[row + k] * B[col + k*m];

		C[cindex] = alpha*sum + C[cindex];
	}
}


// TRMM, returns in C
template <class T, class E, class F>
__global__ void TRMM(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T>* C) {
	int row = get_thread_idx();
	int col = get_thread_idy();
	if (row < n && col < m) {

    int cindex = row*m + col;
    row = row*p;
		CudaInterval<T> sum(0.0);
		for(int k = 0; k < p; ++k)
			sum = sum + A[row + k] * B[col + k*m];

		C[cindex] = alpha*sum + C[cindex];
	}
}


template <class T, class E, class F>
__global__ void GEMM_SHARED(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T>* C) {
  int tx = threadIdx.x, ty = threadIdx.y;
	int row = blockIdx.y*BLOCK_Y + threadIdx.y;
	int col = blockIdx.x*BLOCK_X + threadIdx.x;

  __shared__ CudaInterval<T> AS[BLOCK_X][BLOCK_Y], BS[BLOCK_X][BLOCK_Y];

  CudaInterval<T> sum = 0.0;
  __syncthreads();

    for (int j = 0; j < (p-1)/BLOCK_Y+1; ++j) {

      if (row < n && (j*BLOCK_Y + tx) < p)
        AS[ty][tx] = A[row*p + j*BLOCK_Y + tx];
      else AS[ty][tx] = 0.0;

      if (col < m && (j*BLOCK_X + ty) < p)
        BS[ty][tx] = B[(j*BLOCK_X + ty)*m + col];
      else BS[ty][tx] = 0.0;

      __syncthreads();

		  for(int k = 0; k < BLOCK_X; ++k)
			  sum = sum + AS[ty][k] * BS[k][tx];

      __syncthreads();
    }

	if (row < n && col < m) {
    int cindex = row*m + col;
		C[cindex] = alpha*sum + C[cindex];
	}
}


} // namespace ilag

#endif



