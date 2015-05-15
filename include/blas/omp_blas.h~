
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef BLAS_H
#define BLAS_H

#include <omp.h>
#include <math.h>
#include "aux/trian_index.h"
#include "interval/interval_lib.h"

//TODO: asum, dot, norm2

namespace intlag {
namespace omp {


template <class T>
void sum(int n, Interval<T> const* x, Interval<T>* y) {

	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		y[j] = x[j]+y[j];
}


template <class T>
void mult(int n, Interval<T> const* x, Interval<T>* y) {

	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		y[j] *= x[j];
}


template <class T>
void trans(int n, int m, Interval<T>* A) {

  Interval<T> *B = new Interval<T>[n*m];
	#pragma omp parallel for schedule(dynamic, 500)
	for (int i = 0; i < n; ++i)
	  for (int j = 0; j < m; ++j)
      B[j*n+i] = A[i*m+j];

  acopy(n*m, B, A);
}

template <class T>
void diagonal_unit(int n, int m, Interval<T>* A) {

  int N = n*m;
	#pragma omp parallel for schedule(dynamic, 500)
	for (int i = 0; i < N; i+=m+1)
    A[i] = 1.0;
}


/******************************************************* BLAS 1 ***************/


template<class T, class U>
void acopy(int n, Interval<T> const *x, Interval<U> *y) {

	#pragma omp parallel for schedule(dynamic, 500)
  for(int i = 0; i < n; ++i)
	  y[i] = x[i];
}


template<class T>
void swap(int n, Interval<T> *x, Interval<T> *y) {

	#pragma omp parallel for schedule(dynamic, 500)
  for(int i = 0; i < n; ++i)
	  std::swap(x[i], y[i]);
}


template <class T, class F>
void scal(int n, F alpha, Interval<T>* x) {

	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		x[j] = x[j]*alpha;
}


template<class T, class F>
void axpy(int n, F alpha, Interval<T>* x, Interval<T>* y) {

	#pragma omp parallel for schedule(dynamic, 500)
	for (int j = 0; j < n; ++j)
		y[j] = y[j] + alpha*x[j];
}


// HACK: OpenMP implementation uses indirect reduction since intervals are not considered scalar by OpenMP
template <class T>
void asum(int n, Interval<T>* r, Interval<T>* x) {
	T r1 = 0;
	T r2 = 0;
	#pragma omp parallel for reduction(+:r1, r2)
	for (int j = 0; j < n; ++j){
		Interval<T> r(r1, r2);
		r = r + abs(x[j]);
		r1 = r.inf();
		r2 = r.sup();
	}
	*r = Interval<T>(r1, r2);
}


template <class T>
void dot(int n, Interval<T>* r, Interval<T>* x, Interval<T>* y) {
	T r1 = 0;
	T r2 = 0;
	#pragma omp parallel for reduction(+:r1, r2)
	for (int j = 0; j < n; ++j){
		Interval<T> r(r1, r2);
		r = r + abs(x[j])*abs(y[j]);
		r1 = r.inf();
		r2 = r.sup();
	}
	*r = Interval<T>(r1, r2);
}


template <class T>
void norm2(int n, Interval<T>* r, Interval<T>* x) {
	T r1 = 0;
	T r2 = 0;
	#pragma omp parallel for reduction(+:r1, r2)
	for (int j = 0; j < n; ++j){
		Interval<T> r(r1, r2);
		r = r + abs(x[j])*abs(x[j]);
		r1 = r.inf();
		r2 = r.sup();
	}
	*r = sqrt( Interval<T>(r1, r2) );
}


template <class T, class U, class F>
void rot(int n, Interval<T>* x, U* y, F c, F s){

  #pragma omp parallel for schedule(dynamic, 500)
	for (int i = 0; i < n; ++i) {
    Interval<T> xi = x[i], yi = y[i];
    x[i]  = c*xi + s*yi;
    y[i] = c*yi - s*xi;
  }
}


template <class T, class F>
void rotm(int n, Interval<T>* x, Interval<T>* y, F const *h){

  F h11, h12, h21, h22;

  switch((int)h[0]){
    case -1:
      h11 = h[1];
      h21 = h[2];
      h12 = h[3];
      h22 = h[4];

      #pragma omp parallel for schedule(dynamic, 500)
	    for (int i = 0; i < n; ++i) {
        Interval<T> xi = x[i], yi = y[i];
        x[i] = h11*xi + h12*yi;
        y[i] = h21*xi + h22*yi;
      }

      break;
    case 0:
      h21 = h[2];
      h12 = h[3];

      #pragma omp parallel for schedule(dynamic, 500)
	    for (int i = 0; i < n; ++i) {
        Interval<T> xi = x[i], yi = y[i];
        x[i] = xi + h12*yi;
        y[i] = h21*xi + yi;
      }

      break;
    case 1:
      h11 = h[1];
      h22 = h[4];

      #pragma omp parallel for schedule(dynamic, 500)
	    for (int i = 0; i < n; ++i) {
        Interval<T> xi = x[i], yi = y[i];
        x[i] = h11*xi + yi;
        y[i] = h22*yi - xi;
      }

      break;
    case -2:
      return;
    default: { abort(); }
  }
}


/********************************************************* BLAS 2 *************/


template <class T, class F>
void ger(int n, int m, F alpha, Interval<T> *x, Interval<T> *y, Interval<T> *A) {

  if (alpha == 0.0)
    return;

	#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    Interval<T> xline = alpha*x[i];
    for (int j = 0; j < m; ++j)
      A[i*m+j] += xline*y[j];
  }
}


template <class T, class F>
void syr(char uplo, int n, F alpha, Interval<T> *x, Interval<T> *A) {

  if (alpha == 0.0)
    return;

  if(uplo == 'U' || uplo == 'u'){

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      for (int j = i; j < n; j++)
        A[i*n+j] += xline*x[j];
    }

  }

  else if(uplo == 'L' || uplo == 'l'){

	  #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      for (int j = 0; j <= i; j++)
        A[i*n+j] += xline*x[j];
    }

  }
  else { abort(); }
}


template <class T, class F>
void syr2(char uplo, int n, F alpha, Interval<T> *x, Interval<T> *y, Interval<T> *A) {

  if (alpha == 0.0)
    return;

  if(uplo == 'U' || uplo == 'u'){

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      Interval<T> yline = alpha*y[i];
      for (int j = i; j < n; j++)
        A[i*n+j] += xline*y[j] + yline*x[j];
    }

  }

  else if(uplo == 'L' || uplo == 'l'){

  	#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      Interval<T> yline = alpha*y[i];
      for (int j = 0; j <= i; j++)
        A[i*n+j] += xline*y[j] + yline*x[j];
    }

  }
  else { abort(); }
}


template <class T, class F>
void spr(char uplo, int n, F alpha, Interval<T> *x, Interval<T> *A) {

  if (alpha == 0.0)
    return;

  if(uplo == 'U' || uplo == 'u'){

	  #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      for (int j = i; j < n; j++)
        A[INDEX_TRIAN_UP(n, i, j)] += xline*x[j];
    }

  }

  else if(uplo == 'L' || uplo == 'l'){

  	#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      for (int j = 0; j <= i; j++)
        A[INDEX_TRIAN_DOWN(n, i, j)] += xline*x[j];
    }

  }
  else { abort(); }
}


template <class T, class F>
void spr2(char uplo, int n, F alpha, Interval<T> *x, Interval<T> *y, Interval<T> *A) {

  if (alpha == 0.0)
    return;

  if(uplo == 'U' || uplo == 'u'){

  	#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Interval<T> xline = alpha*x[i];
      Interval<T> yline = alpha*y[i];
      for (int j = i; j < n; j++)
        A[INDEX_TRIAN_UP(n, i, j)] += xline*y[j] + yline*x[j];
    }

  }

  else if(uplo == 'L' || uplo == 'l'){

  	#pragma omp parallel for
    for (int i = 0; i <= n; i++) {
      Interval<T> xline = alpha*x[i];
      Interval<T> yline = alpha*y[i];
      for (int j = 0; j <= i; j++)
        A[INDEX_TRIAN_DOWN(n, i, j)] += xline*y[j] + yline*x[j];
    }

  }
  else { abort(); }
}


template <class T, class E, class F>
void gemv(int n, int m, E alpha, F beta, Interval<T> *A, Interval<T> *x, Interval<T> *y) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n, beta, y);

  if (alpha == 0.0)
    return;

  #pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		Interval<T> sum = 0.0;
		for (int j = 0; j < m; ++j)
			sum += A[i*m + j]*x[j];
		y[i] += alpha*sum;
	}
}


template <class T, class F, class E>
void gbmv(int n, int m, int kl, int ku, F alpha, E beta, Interval<T> *A, Interval<T> *x, Interval<T> *y) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n, beta, y);

  if (alpha == 0.0)
    return;

  int M = kl+ku+1;

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
		Interval<T> sum = 0.0;
    const int j_min = std::max(i-kl, 0);
    const int j_max = std::min(m-1, i+ku);
    for (int j = j_min; j <= j_max; j++) {
      sum += x[j] * A[i*M+kl+j-i];
    }
		y[i] += alpha*sum;
  }
}


template <class T, class E, class F>
void symv(char uplo, int n, E alpha, F beta, Interval<T> *A, Interval<T> *x, Interval<T> *y) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n, beta, y);

  if (alpha == 0.0)
    return;

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      Interval<T> xline = alpha * x[i];
      Interval<T> sum = 0.0;
      y[i] += xline * A[i*n+i]; // diagonal terms
      for (int j = i+1; j < n; ++j) {
        Interval<T> Aij = A[i*n+j];
        sum += x[j] * Aij;  // postdiagonal terms
        y[j] += xline * Aij; // prediagonal terms
      }
      y[i] += alpha*sum;
    }

  }
  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
    for (int i = n-1; i >= 0; --i) {
      Interval<T> xline = alpha * x[i];
      Interval<T> sum = 0.0;
      y[i] += xline * A[i*n+i]; // diagonal terms
      for (int j = 0; j < i; j++) {
        Interval<T> Aij = A[i*n+j];
        sum += x[j] * Aij;  // postdiagonal terms
        y[j] += xline * Aij; // prediagonal terms
      }
      y[i] += alpha*sum;
    }

  }
  else { abort(); }
}


template <class T, class E, class F>
void sbmv(char uplo, int n, int k, E alpha, F beta, Interval<T> *A, Interval<T> *x, Interval<T> *y) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n, beta, y);

  if (alpha == 0.0)
    return;

  int N = k+1;

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      Interval<T> xline = alpha * x[i];
		  Interval<T> sum = 0.0;
      const int j_max = std::min(n-1, i+k);
      y[i] += xline * A[i*N]; // diagonal terms
      for (int j = i+1; j <= j_max; ++j) {
				Interval<T> Aij = A[i*N+j-i];// [i][j-i]
        sum += x[j] * Aij;  // postdiagonal terms
        y[j] += xline * Aij; // prediagonal terms
      }
      y[i] += alpha*sum;
    }

  }

  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      Interval<T> xline = alpha * x[i];
		  Interval<T> sum = 0.0;
      y[i] += xline * A[i*N+k]; // diagonal terms
      const int j_min = std::max(0, i-k);
      for (int j = j_min; j < i; ++j) {
				Interval<T> Aij = A[i*N+k+j-i]; // [i][k+j-i]
        sum += x[j] * Aij;  // postdiagonal terms
        y[j] += xline * Aij; // prediagonal terms
      }
      y[i] += alpha*sum;
    }

  }
  else { abort(); }
}


template <class T, class E, class F>
void spmv(char uplo, int n, E alpha, F beta, Interval<T> *A, Interval<T> *x, Interval<T> *y) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n, beta, y);

  if (alpha == 0.0)
    return;

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      Interval<T> xline = alpha * x[i];
      Interval<T> sum = 0.0;
      y[i] += xline * A[INDEX_TRIAN_UP(n, i, i)]; // diagonal terms
      for (int j = i+1; j < n; ++j) {
				Interval<T> A_UP = A[INDEX_TRIAN_UP(n, i, j)];
        sum += x[j] * A_UP;  // postdiagonal terms
        y[j] += xline * A_UP; // prediagonal terms
      }
      y[i] += alpha*sum;
    }

  }

  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      Interval<T> xline = alpha * x[i];
      Interval<T> sum = 0.0;
      y[i] += xline * A[INDEX_TRIAN_DOWN(n,i,i)]; // diagonal terms
      for (int j = 0; j < i; j++) {
				Interval<T> A_DOWN = A[INDEX_TRIAN_DOWN(n,i,j)];
        sum += x[j] * A_DOWN;  // postdiagonal terms
        y[j] += xline * A_DOWN; // prediagonal terms
      }
      y[i] += alpha*sum;
    }

  }
  else { abort(); }
}


template <class T>
void trmv(char uplo, int n, Interval<T> *A, Interval<T> *x) {

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
			Interval<T> sum = 0.0;
      for (int j = i; j < n; ++j)
        sum += x[j] * A[i*n+j];
			x[i] = sum;
		}

  }
  else if(uplo == 'L' || uplo == 'l'){

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
			Interval<T> sum = 0.0;
      for (int j = 0; j <= i; ++j)
        sum += x[j] * A[i*n+j];
			x[i] = sum;
		}

  }
  else { abort(); }
}


template <class T>
void tbmv(char uplo, int n, int k, Interval<T> *A, Interval<T> *x) {

  int N = k+1;

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
		for (int i = 0; i < n; ++i) {
      Interval<T> sum = 0.0;
      int j_max = std::min(n-1, i+k);
      for (int j = i; j <= j_max; j++)
        sum += x[j] * A[i*N+j-i];
      x[i] = sum;
    }

  }
  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
		for (int i = 0; i < n; ++i) {
      Interval<T> sum = x[i]*A[i*N+k]; //diagonal
      int j_min = std::max(i-k,0);
      for (int j = j_min; j < i; j++)
        sum += x[j] * A[i*N+k+j-i];
      x[i] = sum;
    }

  }
  else { abort(); }
}


template <class T>
void tpmv(char uplo, int n, Interval<T> *A, Interval<T> *x) {

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			Interval<T> sum = 0.0;
      for (int j = i; j < n; j++)
        sum += x[j] * A[INDEX_TRIAN_UP(n, i, j)];
      x[i] = sum;
    }

  }
  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			Interval<T> sum = 0.0;
      for (int j = 0; j <= i; ++j)
        sum += x[j] * A[INDEX_TRIAN_DOWN(n, i, j)];
      x[i] = sum;
    }

  }
  else { abort(); }
}


template <class T>
void trsv(char uplo, int n, Interval<T> *A, Interval<T> *x) {

  if(uplo == 'U' || uplo == 'u') {

		int i = n-1;
    x[i] = x[i] / A[i*n+i];

    #pragma omp parallel for
    for (int i = n-2; i >= 0; --i) {
      Interval<T> sum = x[i];
      for (int j = i+1; j < n; j++)
        sum -= A[i*n+j] * x[j];
      x[i] = sum / A[i*n+i];
    }

  }
  else if(uplo == 'L' || uplo == 'l') {

    x[0] = x[0] / A[0];

    #pragma omp parallel for
    for (int i = 1; i < n; ++i) {
      Interval<T> sum = x[i];
      for (int j = 0; j < i; j++)
        sum -= A[i*n+j] * x[j];
      x[i] = sum / A[i*n+i];
    }

  }
  else { abort(); }
}


template <class T>
void tbsv(char uplo, int n, int k, Interval<T> *A, Interval<T> *x) {


  int N = k+1;

  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
    for (int i = n-1; i >= 0; --i) {
      Interval<T> sum = x[i];
      int j_max = std::min(n-1, i+k);
      for (int j = i+1; j <= j_max; j++)
        sum -= A[i*N+j-i] * x[j];
      x[i] = sum / A[i*N];
    }

  }
  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
		for (int i = 0; i < n; i++) {
      Interval<T> sum = x[i];
      int j_min = std::max(i-k, 0);
      for (int j = j_min; j < i; j++)
        sum -= A[i*N+k+j-i] * x[j];
      x[i] = sum / A[i*N+k];
    }

  }
  else { abort(); }
}


template <class T>
void tpsv(char uplo, int n, Interval<T> *A, Interval<T> *x) {

  if(uplo == 'U' || uplo == 'u') {

		int i = n-1;
    x[i] = x[i] / A[INDEX_TRIAN_UP(n, i, i)];

    #pragma omp parallel for
    for (int i = n-2; i >= 0 ; --i) {
      Interval<T> sum = x[i];
      for (int j = i+1; j < n; j++)
        sum -= A[INDEX_TRIAN_UP(n, i, j)] * x[j];
      x[i] = sum / A[INDEX_TRIAN_UP(n, i, i)];
    }

  }
  else if(uplo == 'L' || uplo == 'l') {

    x[0] = x[0] / A[0];

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      Interval<T> sum = x[i];
      for (int j = 0; j < i; j++)
        sum -= A[INDEX_TRIAN_DOWN(n, i, j)] * x[j];
      x[i] = sum / A[INDEX_TRIAN_DOWN(n, i, i)];
    }

  }
  else { abort(); }
}


/********************************************************* BLAS 3 *************/


template<class T, class F>
void gemm(int n, int m, int p, F alpha, F beta, Interval<T> *A, Interval<T> *B, Interval<T> *C) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n*m, beta, C);

  if (alpha == 0.0)
    return;


  #pragma omp parallel for
	for (int i = 0; i < n; ++i) 
		for (int j = 0; j < m; ++j) {
			Interval<T> sum = 0.0;
			for (int k = 0; k < p; ++k)
				sum += A[i*p+k]*B[k*m+j];
			C[i*m+j] += alpha*sum;
		}
}


template <class T, class E, class F>
void symm(char side, char uplo, int n, int m, E alpha, F beta, Interval<T> *A, Interval<T> *B, Interval<T> *C) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0)
    omp::scal(n*m, beta, C);

  if (alpha == 0.0)
    return;


	if(side == 'L' || side == 'l') {

  	if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
  	    for (int j = 0; j < m; j++) {
  	      Interval<T> bline = alpha * B[i*m+j];
  	      Interval<T> sum = 0.0;
  	      for (int k = i + 1; k < n; k++) {
  	        Interval<T> Aik = A[i*n+k];
  	        C[k*m+j] += bline * Aik;
  	        sum += B[k*m+j] * Aik;
  	      }
  	      C[i*m+j] += bline*A[i*n+i] + alpha*sum;
  	    }
  	  }

  	}
  	else if(uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
	      for (int j = 0; j < m; j++) {
	        Interval<T> bline = alpha * B[i*m+j];
	        Interval<T> sum = 0.0;
	        for (int k = 0; k < i; k++) {
	          Interval<T> Aik = A[i*n+k];
	          C[k*m+j] += bline * Aik;
	          sum += B[k*m+j] * Aik;
	        }
	        C[i*m+j] += bline * A[i*n+i] + alpha * sum;
	      }
	    }

		}
		else { abort(); }

	}

	else if(side == 'R' || side == 'r') {

		if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
  	    for (int j = 0; j < m; j++) {
  	      Interval<T> bline = alpha * B[i*m+j];
  	      Interval<T> sum = 0.0;
  	      for (int k = j + 1; k < m; k++) {
  	        Interval<T> Ajk = A[j*m+k];
  	        C[i*m+k] += bline * Ajk;
  	        sum += B[i*m+k] * Ajk;
  	      }
  	      C[i*m+j] += bline * A[j*m+j] + alpha * sum;
  	    }
  	  }

  	}
  	else if(uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
	      for (int j = 0; j < m; j++) {
	        Interval<T> bline = alpha * B[i*m+j];
	        Interval<T> sum = 0.0;
	        for (int k = 0; k < j; k++) {
	          Interval<T> Ajk = A[j*m+k];
	          C[i*m+k] += bline * Ajk;
	          sum += B[i*m+k] * Ajk;
	        }
	        C[i*m+j] += bline * A[j*m+j] + alpha * sum;
	      }
	    }

		}
		else { abort(); }

	}
	else { abort(); }

}


template <class T, class E, class F>
void syrk(char uplo, int n, int k, E alpha, F beta, Interval<T> *A, Interval<T> *C) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0) {
    if(uplo == 'U' || uplo == 'u') {
      #pragma omp parallel for
      for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
          C[i*n+j] *= beta;
    }
    else if (uplo == 'L' || uplo == 'l') {
      #pragma omp parallel for
      for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
          C[i*n+j] *= beta;
    }
    else { abort(); }
  }

  if (alpha == 0.0)
    return;


  if(uplo == 'U' || uplo == 'u') {

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        Interval<T> sum = 0.0;
        for (int ki = 0; ki < k; ki++)
          sum += A[i+ki] * A[j+ki];
        C[i*n+j] += alpha * sum;
      }
    }
  }

  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        Interval<T> sum = 0.0;
        for (int ki = 0; ki < k; ki++)
          sum += A[i*k+ki] * A[j*k+ki];
        C[i*n+j] += alpha * sum;
      }
    }
  }
  else { abort(); }

}


template <class T, class E, class F>
void syr2k(char uplo, int n, int k, E alpha, F beta, Interval<T> *A, Interval<T> *B, Interval<T> *C) {

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (beta != 1.0) {
    if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
      for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
          C[i*n+j] *= beta;
    }
    else if (uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
      for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
          C[i*n+j] *= beta;
    }
    else { abort(); }
  }

  if (alpha == 0.0)
    return;


  if(uplo == 'U' || uplo == 'u'){

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        Interval<T> sum = 0.0;
        for (int ki = 0; ki < k; ki++)
          sum += A[i*k+ki] * B[j*k+ki] + B[i*k+ki]*A[j*k+ki];
        C[i*n+j] += alpha * sum;
      }
    }

  }

  else if(uplo == 'L' || uplo == 'l') {

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        Interval<T> sum = 0.0;
        for (int ki = 0; ki < k; ki++)
          sum += A[i*k+ki] * B[j*k+ki] + B[i*k+ki]*A[j*k+ki];
        C[i*n+j] += alpha * sum;
      }
    }
  }

  else { abort(); }
}


template <class T, class F>
void trmm(char side, char uplo, int n, int m, F alpha, Interval<T> *A, Interval<T> *B) {

  if (alpha == 0.0)
    return;

	if(side == 'L' || side == 'l') {

  	if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
    	for (int i = 0; i < n; ++i) {
     		for (int j = 0; j < m; ++j) {
      	  Interval<T> sum = 0.0;
      	  for (int k = i; k < n; ++k)
      	    sum += A[i*n+k] * B[k*m+j];
      	  B[i*m+j] = alpha * sum;
      	}
   	  }

  	}
  	else if(uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
    	for (int i = n-1; i >= 0; --i) {
     		for (int j = 0; j < m; ++j) {
      	  Interval<T> sum = 0.0;
      	  for (int k = 0; k <= i; ++k)
      	    sum += A[i*n+k] * B[k*m+j];
      	  B[i*m+j] = alpha * sum;
      	}
    	}

		}
		else { abort(); }

	}

	else if(side == 'R' || side == 'r') {

  	if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
		  for (int i = 0; i < n; ++i) {
     		for (int j = m-1; j >= 0; --j) {
      	  Interval<T> sum = 0.0;
      	  for (int k = 0; k <= j; ++k)
      	    sum += B[i*m+k] * A[k*m+j];
      	  B[i*m+j] = alpha * sum;
      	}
   	  }

  	}
  	else if(uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
		  for (int i = 0; i < n; i++) {
     		for (int j = 0; j < m; j++) {
      	  Interval<T> sum = 0.0;
      	  for (int k = j; k < m; k++)
      	    sum += B[i*m+k] * A[k*m+j];
      	  B[i*m+j] = alpha * sum;
      	}
   	  }

		}
		else { abort(); }

	}
	else { abort(); }

}


template <class T, class F>
void trsm(char side, char uplo, int n, int m, F alpha, Interval<T> *A, Interval<T> *B) {

  if (alpha != 1.0)
    omp::scal(n*m, alpha, B);

	if(side == 'L' || side == 'l') {

  	if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
		  for (int i = n-1; i >= 0; --i) {
		    Interval<T> Aii = A[i*n+i];
		    for (int j = 0; j < m; j++)
		      B[i*m+j] /= Aii;

		    for (int k = 0; k < i; k++) {
		      Interval<T> Aki = A[k*n+i];
		      for (int j = 0; j < m; j++)
		        B[k*m+j] -= Aki * B[i*m+j];
		    }
		  }

  	}
  	else if(uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
		      Interval<T> Aii = A[i*n+i];
		      for (int j = 0; j < m; j++)
		        B[i*m+j] /= Aii;

		    for (int k = i+1; k < n; k++) {
		      Interval<T> Aki = A[k*n+i];
		      for (int j = 0; j < m; j++)
		        B[k*m+j] -= Aki * B[i*m+j];
		    }
		  }

		}
		else { abort(); }

	}

	else if(side == 'R' || side == 'r') {

  	if(uplo == 'U' || uplo == 'u') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
		    for (int j = 0; j < m; j++) {
		      B[i*m+j] /= A[j*m+j];
		      Interval<T> Bij = B[i*m+j];
		      for (int k = j+1; k < m; k++)
		        B[i*m+k] -= Bij*A[j*m+k];
		    }
		  }

  	}
  	else if(uplo == 'L' || uplo == 'l') {

      #pragma omp parallel for
			for (int i = 0; i < n; i++) {
		    for (int j = m-1; j >= 0; --j) {
		        B[i*m+j] /= A[j*m+j];
		        Interval<T> Bij = B[i*m+j];
		        for (int k = 0; k < j; k++)
		          B[i*m+k] -= Bij*A[j*m+k];
		    }
		  }

		}
		else { abort(); }

	}
	else { abort(); }

}




} // namespace omp
} // namespace intlag

#endif



