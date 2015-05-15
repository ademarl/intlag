
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BLAS_FUNCTIONS_H
#define CUDA_BLAS_FUNCTIONS_H


#include <stdexcept>

#include "cuda_blas_kernels.h"

#include "aux/cuda_grid.h"
#include "aux/cuda_error.h"
#include "aux/device_data.h"


namespace intlag {

/******************************************* Auxiliar *************************/

template <class T>
void trans(int n, int m, CudaInterval<T> const *A, CudaInterval<T> *B) {

  for(int i = 0; i < n; ++i)
    for(int j = 0; j < m; ++j)
      B[n*j+i] = A[m*i+j];
}


// Loops through kernels which calculate the sum of adjacent pairs, storing
//them on temporary memory, then swaps the references to these memory locations
template <class T> inline __host__
void asum_loop(int N, DeviceData< CudaInterval<T> >& x) {
	DeviceData< CudaInterval<T> > y((N+1)/2);

	while(N != 1) {
		int n = (N+1)/2;
		ASUM<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x.data(), y.data(), N);
	  CHECKED_CALL( cudaDeviceSynchronize() );
		swapByReference(x, y);
		N = n;
	}
}


template <class T> inline __host__
void asum_loop(CudaInterval<T> *ret, int N, CudaInterval<T> const *a) {
	DeviceData< CudaInterval<T> > x(N), y((N+1)/2);
  COPY<<<CudaGrid::blocks(N), CudaGrid::threads()>>>(a, x.data(), N);
  CHECKED_CALL( cudaDeviceSynchronize() );

	while(N != 1) {
		int n = (N+1)/2;
		ASUM<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x.data(), y.data(), N);
	  CHECKED_CALL( cudaDeviceSynchronize() );
		swapByReference(x, y);
		N = n;
	}
  //CHECKED_CALL( cudaDeviceSynchronize() ); // need this?
  COPY<<<1,1>>>(x.data(), ret, 1);
  CHECKED_CALL( cudaDeviceSynchronize() );
}


/******************************************************** General *************/

class CudaGeneral {
  public:


    /**************************************************** BLAS level 1 ********/
    template <class T, class F> inline __host__
    static void scal(int n, F alpha, CudaInterval<T> *x) {
      if (alpha == 1.0) return;
	    SCALAR_MULT<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(alpha, x, n);
    }

    template <class T, class F> inline __host__
    static void axpy(int n, F alpha, CudaInterval<T> const *x, CudaInterval<T> *y) {
	    if (alpha == 0.0) return;
	    if (alpha == 1.0)
		    ADD<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, y, n);
	    else
		    AXPY<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(alpha, x, y, n);
    }

    template <class T> inline __host__
    static void asum(int n, CudaInterval<T> *ret, CudaInterval<T> const *x) {
	    asum_loop(ret, n, x);
    }

    // need const
    template <class T> inline __host__
    static void dot(int n, CudaInterval<T> *ret, CudaInterval<T> const *x, CudaInterval<T> *y) {
      // Calculate Hadamard product
	    NORMMULT<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, y, n);
	    CHECKED_CALL( cudaDeviceSynchronize() );
      // Then sum all elements
	    asum_loop(ret, n, y);
    }

    // need const
    template <class T> inline __host__
    static void norm2(int n, CudaInterval<T> *ret, CudaInterval<T> *x) {
      // Calculate Hadamard product
	    NORMMULT<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, x, n);
	    CHECKED_CALL( cudaDeviceSynchronize() );
      // Then sum all elements
	    asum_loop(ret, n, x);
      // Then take the squareroot
	    SQRT<<<1, 1>>>(ret, ret);
	    CHECKED_CALL( cudaDeviceSynchronize() );
    }


    template<class T, class F> inline __host__
    static void rot(int n, CudaInterval<T>* x, CudaInterval<F>* y, F c, F s) {
      ROT<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, y, c, s, n);
    }


    template<class T, class F> inline __host__
    static void rotm(int n, CudaInterval<T>* x, CudaInterval<T>* y, F* H) {
      int h = (int)H[0];
      if (h == 2) return;
      else if (h == 1)
        ROTM1<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, y, H[1], H[4], n);
      else if (h == 0)
        ROTM0<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, y, H[2], H[3], n);
      else if (h == -1)
        ROTMM1<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(x, y, H[1], H[2], H[3], H[4], n);
      else { abort(); }
    }


    /**************************************************** BLAS level 2 ********/
    template <class T, class F> inline __host__
    static void ger(int n, int m, F alpha, CudaInterval<T> *x, CudaInterval<T> *y, CudaInterval<T> *A) {
      if (alpha == 0.0)
        return;
      GER<<<CudaGrid::blocks2(n,m), CudaGrid::threads2()>>>(n, m, alpha, x, y, A);
    }



    template <class T, class E, class F> inline __host__
    static void gemv(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {
	    if (alpha == 0)
		    scal(n, beta, y);
	    GEMV<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(n, m, alpha, beta, A, x, y);
    }


    /**************************************************** BLAS level 3 ********/
    template <class T, class E, class F> inline __host__
    static void gemm(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T> *C) {
      int N = n*m;

      // if alpha = 0, simply do y = beta*y
      scal(N, beta, C);
	    if (alpha == 0.0)
        return;
	    GEMM<<<CudaGrid::blocks2(N,m), CudaGrid::threads2()>>>(n, m, p, alpha, beta, A, B, C);
    }
};

/******************************************************** ~General ************/


/******************************************************** GeneralManaged ******/

class CudaGeneralManaged {
  public:


    /**************************************************** BLAS level 1 ********/
    template <class T, class F> inline __host__
    static void scal(int n, F alpha, CudaInterval<T> *x) {
      if (alpha == 1.0) return;

      DeviceData< CudaInterval<T> > d_x(n, x);

	    CudaGeneral::scal(n, alpha, d_x.data());

	    d_x.toHost(x);
    }

    template <class T, class F> inline __host__
    static void axpy(int n, F alpha, CudaInterval<T> const *x, CudaInterval<T> *y) {
	    if (alpha == 0.0) return;

	    DeviceData< CudaInterval<T> > d_x(n, x), d_y(n, y);

	    CudaGeneral::axpy(n, alpha, d_x.data(), d_y.data());

	    d_y.toHost(y);
    }

    template <class T> inline __host__
    static void asum(int n, CudaInterval<T> *ret, CudaInterval<T> const *x) {
	    DeviceData< CudaInterval<T> > d_x(n, x);

	    asum_loop(n, d_x);

	    d_x.toHost(ret, 1);
    }

    template <class T> inline __host__
    static void dot(int n, CudaInterval<T> *ret, CudaInterval<T> const *x, CudaInterval<T> const *y) {
	    DeviceData< CudaInterval<T> > d_x(n, x), d_y(n, y);

	    CudaGeneral::dot(n, d_y.data(), d_x.data(), d_y.data());

	    d_y.toHost(ret, 1);
    }

    template <class T> inline __host__
    static void norm2(int n, CudaInterval<T> *ret, CudaInterval<T> const *x) {
	    DeviceData< CudaInterval<T> > d_x(n, x);
	    
      CudaGeneral::norm2(n, d_x.data(), d_x.data());

	    d_x.toHost(ret, 1);
    }

    template <class T, class F> inline __host__
    static void rot(int n, CudaInterval<T> *x, CudaInterval<T> *y, F const c, F const s) {
      DeviceData< CudaInterval<T> > d_x(n, x), d_y(n, y);

      ROT<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(d_x.data(), d_y.data(), c, s, n);

      d_x.toHost(x, n);
      d_y.toHost(y, n);
    }


    template<class T, class F> inline __host__
    static void rotm(int n, CudaInterval<T>* x, CudaInterval<T>* y, F* H) {
      int h = H[0];

      if (h == -2) return;      

      DeviceData< CudaInterval<T> > d_x(n, x), d_y(n, y);

      if (h == 1)
        ROTM1<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(d_x.data(), d_y.data(), H[1], H[4], n);
      else if (h == 0)
        ROTM0<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(d_x.data(), d_y.data(), H[2], H[3], n);
      else if (h == -1)
        ROTMM1<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(d_x.data(), d_y.data(), H[1], H[2], H[3], H[4], n);
      else { abort(); }

      d_x.toHost(x, n);
      d_y.toHost(y, n);
    }


    /**************************************************** BLAS level 2 ********/
    template <class T, class F> inline __host__
    static void ger(int n, int m, F alpha, CudaInterval<T> *x, CudaInterval<T> *y, CudaInterval<T> *A) {
      if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_x(n, x), d_y(m, y), d_A(n*m, A);

      GER<<<CudaGrid::blocks2(n,m), CudaGrid::threads2()>>>(n, m, alpha, d_x.data(), d_y.data(), d_A.data());

	    d_A.toHost(A, n*m);
    }


    template <class T, class E, class F> inline __host__
    static void gemv(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {
	    // if alpha = 0, simply do y = beta*y
      scal(n, beta, y);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*m, A), d_x(m, x), d_y(n, y);

	    GEMV<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(n, m, alpha, beta, d_A.data(), d_x.data(), d_y.data());

	    d_y.toHost(y, n);
    }


    /**************************************************** BLAS level 3 ********/
    template <class T, class E, class F> inline __host__
    static void gemm(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T> *C) {
      int N = n*m;

      // if alpha = 0, simply do y = beta*y
      scal(N, beta, C);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*p, A), d_B(p*m, B), d_C(N, C);

	    GEMM<<<CudaGrid::blocks2(n,m), CudaGrid::threads2()>>>(n, m, p, alpha, beta, d_A.data(), d_B.data(), d_C.data());

	    d_C.toHost(C, N);
    }

    template <class T, class E, class F> inline __host__
    static void symm(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T> *C) {
      int N = n*m;

      // if alpha = 0, simply do y = beta*y
      scal(N, beta, C);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*p, A), d_B(p*m, B), d_C(N, C);

	    SYMM<<<CudaGrid::blocks2(n,m), CudaGrid::threads2()>>>(n, m, p, alpha, beta, d_A.data(), d_B.data(), d_C.data());

	    d_C.toHost(C, N);
    }
};

/******************************************************** ~Shared *****/

class CudaShared {
  public:

    template <class T, class E, class F> inline __host__
    static void gemv(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {
      scal(n, beta, y);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*m, A), d_x(m, x), d_y(n, y);

	    GEMV_SHARED<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(n, m, alpha, beta, d_A.data(), d_x.data(), d_y.data());

	    d_y.toHost(y, n);
    }


    template <class T, class E, class F> inline __host__
    static void gemm(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T> *C) {
      int N = n*m;

      // if alpha = 0, simply do y = beta*y
      scal(N, beta, C);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*p, A), d_B(p*m, B), d_C(N, C);

	    GEMM_SHARED<<<CudaGrid::blocks2(n,m), CudaGrid::threads2()>>>(n, m, p, alpha, beta, d_A.data(), d_B.data(), d_C.data());

	    d_C.toHost(C, N);
    }
};


class CudaSharedManaged {
  public:

    template <class T, class E, class F> inline __host__
    static void gemv(int n, int m, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *x, CudaInterval<T> *y) {
      scal(n, beta, y);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*m, A), d_x(m, x), d_y(n, y);

	    GEMV_SHARED<<<CudaGrid::blocks(n), CudaGrid::threads()>>>(n, m, alpha, beta, d_A.data(), d_x.data(), d_y.data());

	    d_y.toHost(y, n);
    }


    template <class T, class E, class F> inline __host__
    static void gemm(int n, int m, int p, E alpha, F beta, CudaInterval<T> const *A, CudaInterval<T> const *B, CudaInterval<T> *C) {
      int N = n*m;

      // if alpha = 0, simply do y = beta*y
      scal(N, beta, C);
	    if (alpha == 0.0)
        return;

	    DeviceData < CudaInterval<T> > d_A(n*p, A), d_B(p*m, B), d_C(N, C);

	    GEMM_SHARED<<<CudaGrid::blocks2(n,m), CudaGrid::threads2()>>>(n, m, p, alpha, beta, d_A.data(), d_B.data(), d_C.data());

	    d_C.toHost(C, N);
    }
};


} // namespace ilag
#endif



