
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef OMP_BENCH_H
#define OMP_BENCH_H


#include <omp.h>

#include "aux/case.h"
#include "aux/reference.h"

#include "../include/blas/serial_blas.h"
#include "../include/blas/omp_blas.h"
#include "gtest/gtest.h"

namespace intlag {
namespace bench {


//----------------------- Fixture --------------------------------------------//

class OMPBench : public BenchTest {
  public:

    OMPBench() {
	    //omp_set_num_threads(16);
      r = Reference::getInstance();
    }
    virtual ~OMPBench() {}

    void SetUp() {}
    void TearDown() {}

    Reference* r;
};


//----------------------- Scal Bench -----------------------------------------//

template <class T>
class OMPBenchScal : public OMPBench  {
   public:

    void begin() {


      n = r->length;
      alpha = r->alpha;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      omp::scal(n, alpha, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    Interval<T> *x, alpha;
};
BENCH_F_F(OMPBench, Scal, OMPBenchScal)


//----------------------- AXPY Bench -----------------------------------------//

template <class T>
class OMPBenchAXPY : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      alpha = r->alpha;
      x = new Interval<T> [n];
      y = new Interval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      omp::axpy(n, alpha, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    Interval<T> *x, *y, alpha;
};
BENCH_F_F(OMPBench, AXPY, OMPBenchAXPY)


//----------------------- ASum Bench -----------------------------------------//

template <class T>
class OMPBenchASum : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      omp::asum(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    Interval<T> *x, ret;
};
BENCH_F_F(OMPBench, Asum, OMPBenchASum)


//----------------------- Dot Bench -----------------------------------------//

template <class T>
class OMPBenchDot : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      y = new Interval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      omp::dot(n, &ret, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    Interval<T> *x, *y, ret;
};
BENCH_F_F(OMPBench, Dot, OMPBenchDot)


//----------------------- Norm2 Bench -----------------------------------------//

template <class T>
class OMPBenchNorm2 : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      omp::norm2(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    Interval<T> *x, ret;
};
BENCH_F_F(OMPBench, Norm2, OMPBenchNorm2)


//----------------------- Rot Bench -----------------------------------------//

template <class T>
class OMPBenchRot : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
      y = new Interval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      omp::rot(n, x, y, 0.5, 0.5);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    Interval<T> *x, *y;
};
BENCH_F_F(OMPBench, Rot, OMPBenchRot)


//----------------------- Rotm Bench -----------------------------------------//

template <class T>
class OMPBenchRotm : public OMPBench  {
   public:

    void begin() {
      h[0] = -1; h[1] = 0.5; h[2] = 0.6; h[3] = 0.7; h[4] = 0.8;
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
      y = new Interval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      omp::rotm(n, x, y, h);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    T h[5];
    Interval<T> *x, *y;
};
BENCH_F_F(OMPBench, Rotm, OMPBenchRotm)


//----------------------- Ger Bench -----------------------------------------//


template <class T>
class OMPBenchGer : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::ger(dim, dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Ger, OMPBenchGer)


//----------------------- Syr Bench -----------------------------------------//


template <class T>
class OMPBenchSyr : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
    }

    void run() {
      omp::syr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(OMPBench, Syr, OMPBenchSyr)


//----------------------- Syr2 Bench -----------------------------------------//


template <class T>
class OMPBenchSyr2 : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::syr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Syr2, OMPBenchSyr2)


//----------------------- Spr Bench -----------------------------------------//


template <class T>
class OMPBenchSpr : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
    }

    void run() {
      omp::spr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(OMPBench, Spr, OMPBenchSpr)


//----------------------- Spr2 Bench -----------------------------------------//


template <class T>
class OMPBenchSpr2 : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::spr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Spr2, OMPBenchSpr2)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class OMPBenchGemv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::gemv(dim, dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Gemv, OMPBenchGemv)


//----------------------- Gbmv Bench -----------------------------------------//


template <class T>
class OMPBenchGbmv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      alpha = r->alpha;
      beta = r->beta;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::gbmv(dim, dim, band, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Gbmv, OMPBenchGbmv)


//----------------------- Symv Bench -----------------------------------------//


template <class T>
class OMPBenchSymv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::symv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Symv, OMPBenchSymv)


//----------------------- Sbmv Bench -----------------------------------------//


template <class T>
class OMPBenchSbmv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      alpha = r->alpha;
      beta = r->beta;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::sbmv('u', dim, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Sbmv, OMPBenchSbmv)


//----------------------- Spmv Bench -----------------------------------------//


template <class T>
class OMPBenchSpmv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      y = new Interval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      omp::spmv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(OMPBench, Spmv, OMPBenchSpmv)


//----------------------- Trmv Bench -----------------------------------------//


template <class T>
class OMPBenchTrmv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
    }

    void run() {
      omp::trmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x;
};
BENCH_F_F(OMPBench, Trmv, OMPBenchTrmv)


//----------------------- Tbmv Bench -----------------------------------------//


template <class T>
class OMPBenchTbmv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
    }

    void run() {
      omp::tbmv('u', dim, band, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    Interval<T> *A, *x;
};
BENCH_F_F(OMPBench, Tbmv, OMPBenchTbmv)


//----------------------- Tpmv Bench -----------------------------------------//


template <class T>
class OMPBenchTpmv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
    }

    void run() {
      omp::tpmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x;
};
BENCH_F_F(OMPBench, Tpmv, OMPBenchTpmv)


//----------------------- Trsv Bench -----------------------------------------//


template <class T>
class OMPBenchTrsv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
      for(int i = 0; i < dim; ++i)
        A[i*dim+i] = intlag::abs(A[i*dim+i]) + 0.1;
    }

    void run() {
      omp::trsv('u', dim, A, x);
    }

    void end() {

      for(int i = 0; i < dim; ++i)
        EXPECT_FALSE(isnan(x[i].inf()));

      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x;
};
BENCH_F_F(OMPBench, Trsv, OMPBenchTrsv)


//----------------------- Tbsv Bench -----------------------------------------//


template <class T>
class OMPBenchTbsv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      band = dim/4;
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
      for(int i = 0; i < dim; ++i)
        A[i*band+i] = intlag::abs(A[i*band+i]) + 0.1;
    }

    void run() {
      omp::tbsv('u', dim, band, A, x);
    }

    void end() {

      for(int i = 0; i < dim; ++i)
        EXPECT_FALSE(isnan(x[i].inf()));

      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    Interval<T> *A, *x;
};
BENCH_F_F(OMPBench, Tbsv, OMPBenchTbsv)


//----------------------- Tpsv Bench -----------------------------------------//


template <class T>
class OMPBenchTpsv : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      A = new Interval<T> [n];
      x = new Interval<T> [dim];
      acopy(n, r->y, A);
      acopy(dim, r->x, x);
      for(int i = 0; i < dim; ++i)
        A[INDEX_TRIAN_UP(dim, i, i)] = intlag::abs(A[INDEX_TRIAN_UP(dim, i, i)]) + 0.1;
    }

    void run() {
      omp::tpsv('u', dim, A, x);
    }

    void end() {

      for(int i = 0; i < dim; ++i)
        EXPECT_FALSE(isnan(x[i].inf()));

      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x;
};
BENCH_F_F(OMPBench, Tpsv, OMPBenchTpsv)


//----------------------- Gemm Bench -----------------------------------------//


template <class T>
class OMPBenchGemm : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      B = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      C = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      omp::gemm(dim, dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, *C, alpha, beta;
};
BENCH_F_F(OMPBench, Gemm, OMPBenchGemm)


//----------------------- Symm Bench -----------------------------------------//


template <class T>
class OMPBenchSymm : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      B = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      C = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      omp::symm('l', 'u', dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, *C, alpha, beta;
};
BENCH_F_F(OMPBench, Symm, OMPBenchSymm)


//----------------------- Syrk Bench -----------------------------------------//


template <class T>
class OMPBenchSyrk : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      B = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
    }

    void run() {
      omp::syrk( 'u', dim, dim, alpha, beta, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, alpha, beta;
};
BENCH_F_F(OMPBench, Syrk, OMPBenchSyrk)


//----------------------- Syr2k Bench -----------------------------------------//


template <class T>
class OMPBenchSyr2k : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      B = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      C = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      omp::syr2k('u', dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, *C, alpha, beta;
};
BENCH_F_F(OMPBench, Syr2k, OMPBenchSyr2k)


//----------------------- Trmm Bench -----------------------------------------//


template <class T>
class OMPBenchTrmm : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      B = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
    }

    void run() {
      omp::trmm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, alpha;
};
BENCH_F_F(OMPBench, Trmm, OMPBenchTrmm)


//----------------------- Trsm Bench -----------------------------------------//


template <class T>
class OMPBenchTrsm : public OMPBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      B = (Interval<T>*) malloc(n*sizeof(Interval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      for(int i = 0; i < dim; ++i)
        A[i*dim+i] = intlag::abs(A[i*dim+i]) + 0.1;
    }

    void run() {
      omp::trsm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {

      for(int i = 0; i < dim*dim; ++i)
        EXPECT_FALSE(isnan(B[i].inf()));

      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, alpha;
};
BENCH_F_F(OMPBench, Trsm, OMPBenchTrsm)



} // namespace bench
} // namespace intlag

#endif



