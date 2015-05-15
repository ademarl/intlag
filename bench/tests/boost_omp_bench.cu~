
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef BOOST_OMP_BENCH_H
#define BOOST_OMP_BENCH_H

#include <omp.h>

#include "boost/numeric/interval.hpp"
using boost::numeric::interval;
using namespace boost::numeric;


#include "aux/case.h"
#include "aux/reference.h"
#include "../include/blas/boost_omp_blas.h"

#include "gtest/gtest.h"



namespace intlag {
namespace bench {

template<class T, class U>
void acopy(int n, intlag::Interval<T> const *x, interval<U> *y) {
		for(int i = 0; i < n; ++i)
			y[i] = interval<T>(x[i].inf(), x[i].sup());
}

//----------------------- Fixture --------------------------------------------//

class BoostOMPBench : public BenchTest {
  public:

    BoostOMPBench() {
	    //omp_set_num_threads(16);
      r = Reference::getInstance();
    }
    virtual ~BoostOMPBench() {}

    void SetUp() {}
    void TearDown() {}

    Reference* r;
};


//----------------------- Scal Bench -----------------------------------------//

template <class T>
class BoostOMPBenchScal : public BoostOMPBench  {
   public:

    void begin() {


      n = r->length;
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      x = new interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      bi_omp::scal(n, alpha, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    interval<T> *x, alpha;
};
BENCH_FD_F(BoostOMPBench, Scal, BoostOMPBenchScal)


//----------------------- AXPY Bench -----------------------------------------//

template <class T>
class BoostOMPBenchAXPY : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      x = new interval<T> [n];
      y = new interval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      bi_omp::axpy(n, alpha, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    interval<T> *x, *y, alpha;
};
BENCH_F_F(BoostOMPBench, AXPY, BoostOMPBenchAXPY)


//----------------------- ASum Bench -----------------------------------------//

template <class T>
class BoostOMPBenchASum : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      bi_omp::asum(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    interval<T> *x, ret;
};
BENCH_F_F(BoostOMPBench, Asum, BoostOMPBenchASum)


//----------------------- Dot Bench -----------------------------------------//

template <class T>
class BoostOMPBenchDot : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      y = new interval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      bi_omp::dot(n, &ret, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    interval<T> *x, *y, ret;
};
BENCH_F_F(BoostOMPBench, Dot, BoostOMPBenchDot)


//----------------------- Norm2 Bench -----------------------------------------//

template <class T>
class BoostOMPBenchNorm2 : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      bi_omp::norm2(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    interval<T> *x, ret;
};
BENCH_F_F(BoostOMPBench, Norm2, BoostOMPBenchNorm2)


//----------------------- Rot Bench -----------------------------------------//

template <class T>
class BoostOMPBenchRot : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
      y = new interval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      bi_omp::rot(n, x, y, interval<T>(0.5), interval<T>(0.5));
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    interval<T> *x, *y;
};
BENCH_F_F(BoostOMPBench, Rot, BoostOMPBenchRot)


//----------------------- Rotm Bench -----------------------------------------//

template <class T>
class BoostOMPBenchRotm : public BoostOMPBench  {
   public:

    void begin() {
      h[0] = -1; h[1] = 0.5; h[2] = 0.6; h[3] = 0.7; h[4] = 0.8;
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
      y = new interval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      bi_omp::rotm(n, x, y, h);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    T h[5];
    interval<T> *x, *y;
};
BENCH_F_F(BoostOMPBench, Rotm, BoostOMPBenchRotm)


//----------------------- Ger Bench -----------------------------------------//


template <class T>
class BoostOMPBenchGer : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::ger(dim, dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Ger, BoostOMPBenchGer)


//----------------------- Syr Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSyr : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
    }

    void run() {
      bi_omp::syr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Syr, BoostOMPBenchSyr)


//----------------------- Syr2 Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSyr2 : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::syr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Syr2, BoostOMPBenchSyr2)


//----------------------- Spr Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSpr : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
    }

    void run() {
      bi_omp::spr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Spr, BoostOMPBenchSpr)


//----------------------- Spr2 Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSpr2 : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::spr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Spr2, BoostOMPBenchSpr2)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchGemv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::gemv(dim, dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Gemv, BoostOMPBenchGemv)


//----------------------- Gbmv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchGbmv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::gbmv(dim, dim, band, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Gbmv, BoostOMPBenchGbmv)


//----------------------- Symv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSymv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::symv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Symv, BoostOMPBenchSymv)


//----------------------- Sbmv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSbmv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::sbmv('u', dim, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Sbmv, BoostOMPBenchSbmv)


//----------------------- Spmv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSpmv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = new interval<T> [n];
      x = new interval<T> [dim];
      y = new interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      bi_omp::spmv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Spmv, BoostOMPBenchSpmv)


//----------------------- Trmv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTrmv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
    }

    void run() {
      bi_omp::trmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostOMPBench, Trmv, BoostOMPBenchTrmv)


//----------------------- Tbmv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTbmv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
    }

    void run() {
      bi_omp::tbmv('u', dim, band, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    interval<T> *A, *x;
};
BENCH_F_F(BoostOMPBench, Tbmv, BoostOMPBenchTbmv)


//----------------------- Tpmv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTpmv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
    }

    void run() {
      bi_omp::tpmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostOMPBench, Tpmv, BoostOMPBenchTpmv)


//----------------------- Trsv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTrsv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
      for(int i = 0; i < dim; ++i)
        A[i*dim+i] = abs(A[i*dim+i]) + interval<T>(0.1);
    }

    void run() {
      bi_omp::trsv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostOMPBench, Trsv, BoostOMPBenchTrsv)


//----------------------- Tbsv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTbsv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
      for(int i = 0; i < dim; ++i)
        A[i*band+i] = abs(A[i*band+i]) + interval<T>(0.1);
    }

    void run() {
      bi_omp::tbsv('u', dim, band, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    interval<T> *A, *x;
};
BENCH_F_F(BoostOMPBench, Tbsv, BoostOMPBenchTbsv)


//----------------------- Tpsv Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTpsv : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new interval<T> [n];
      x = new interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
      for(int i = 0; i < dim; ++i)
        A[INDEX_TRIAN_UP(dim, i, i)] = abs(A[INDEX_TRIAN_UP(dim, i, i)]) + interval<T>(0.1);
    }

    void run() {
      bi_omp::tpsv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostOMPBench, Tpsv, BoostOMPBenchTpsv)


//----------------------- Gemm Bench -----------------------------------------//


template <class T>
class BoostOMPBenchGemm : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = (interval<T>*) malloc(n*sizeof(interval<T>));
      B = (interval<T>*) malloc(n*sizeof(interval<T>));
      C = (interval<T>*) malloc(n*sizeof(interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      bi_omp::gemm(dim, dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, *C, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Gemm, BoostOMPBenchGemm)


//----------------------- Symm Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSymm : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = (interval<T>*) malloc(n*sizeof(interval<T>));
      B = (interval<T>*) malloc(n*sizeof(interval<T>));
      C = (interval<T>*) malloc(n*sizeof(interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      bi_omp::symm('l', 'u', dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, *C, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Symm, BoostOMPBenchSymm)


//----------------------- Syrk Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSyrk : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = (interval<T>*) malloc(n*sizeof(interval<T>));
      B = (interval<T>*) malloc(n*sizeof(interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
    }

    void run() {
      bi_omp::syrk( 'u', dim, dim, alpha, beta, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Syrk, BoostOMPBenchSyrk)


//----------------------- Syr2k Bench -----------------------------------------//


template <class T>
class BoostOMPBenchSyr2k : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      beta = interval<T>((r->beta).inf(), (r->beta).sup());
      A = (interval<T>*) malloc(n*sizeof(interval<T>));
      B = (interval<T>*) malloc(n*sizeof(interval<T>));
      C = (interval<T>*) malloc(n*sizeof(interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      bi_omp::syr2k('u', dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, *C, alpha, beta;
};
BENCH_F_F(BoostOMPBench, Syr2k, BoostOMPBenchSyr2k)


//----------------------- Trmm Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTrmm : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = (interval<T>*) malloc(n*sizeof(interval<T>));
      B = (interval<T>*) malloc(n*sizeof(interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
    }

    void run() {
      bi_omp::trmm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, alpha;
};
BENCH_F_F(BoostOMPBench, Trmm, BoostOMPBenchTrmm)


//----------------------- Trsm Bench -----------------------------------------//


template <class T>
class BoostOMPBenchTrsm : public BoostOMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      A = (interval<T>*) malloc(n*sizeof(interval<T>));
      B = (interval<T>*) malloc(n*sizeof(interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      for(int i = 0; i < dim; ++i)
        A[i*dim+i] = abs(A[i*dim+i]) + interval<T>(0.1);
    }

    void run() {
      bi_omp::trsm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, alpha;
};
BENCH_F_F(BoostOMPBench, Trsm, BoostOMPBenchTrsm)



} // namespace bench
} // namespace intlag

#endif



