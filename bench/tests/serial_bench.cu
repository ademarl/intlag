
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef SERIAL_BENCH_H
#define SERIAL_BENCH_H


#include "aux/case.h"
#include "aux/reference.h"

#include "../include/blas/serial_blas.h"

#include "gtest/gtest.h"
#include "aux/test_interval.h"


namespace intlag {
namespace bench {


//----------------------- Fixture --------------------------------------------//

class SerialBench : public BenchTest {
  public:

    SerialBench() {
      r = Reference::getInstance();
    }
    virtual ~SerialBench() {}

    void SetUp() {}
    void TearDown() {}

    Reference* r;
};


//----------------------- Scal Bench -----------------------------------------//

template <class T>
class SerialBenchScal : public SerialBench  {
   public:

    void begin() {
      n = r->length;
      alpha = r->alpha;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      scal(n, alpha, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    Interval<T> *x, alpha;
};
BENCH_FD_F(SerialBench, Scal, SerialBenchScal)


//----------------------- AXPY Bench -----------------------------------------//

template <class T>
class SerialBenchAXPY : public SerialBench  {
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
      axpy(n, alpha, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    Interval<T> *x, *y, alpha;
};
BENCH_FD_F(SerialBench, AXPY, SerialBenchAXPY)


//----------------------- ASum Bench -----------------------------------------//

template <class T>
class SerialBenchASum : public SerialBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      asum(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    Interval<T> *x, ret;
};
BENCH_F_F(SerialBench, Asum, SerialBenchASum)


//----------------------- Dot Bench -----------------------------------------//

template <class T>
class SerialBenchDot : public SerialBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      y = new Interval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      dot(n, &ret, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    Interval<T> *x, *y, ret;
};
BENCH_F_F(SerialBench, Dot, SerialBenchDot)


//----------------------- Norm2 Bench -----------------------------------------//

template <class T>
class SerialBenchNorm2 : public SerialBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      norm2(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    Interval<T> *x, ret;
};
BENCH_F_F(SerialBench, Norm2, SerialBenchNorm2)


//----------------------- Rot Bench -----------------------------------------//

template <class T>
class SerialBenchRot : public SerialBench  {
   public:

    void begin() {
      n = r->length;
      x = new Interval<T> [n];
      acopy(n, r->x, x);
      y = new Interval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      rot(n, x, y, 0.5, 0.5);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    Interval<T> *x, *y;
};
BENCH_F_F(SerialBench, Rot, SerialBenchRot)


//----------------------- Rotm Bench -----------------------------------------//

template <class T>
class SerialBenchRotm : public SerialBench  {
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
      rotm(n, x, y, h);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    T h[5];
    Interval<T> *x, *y;
};
BENCH_F_F(SerialBench, Rotm, SerialBenchRotm)


//----------------------- Ger Bench -----------------------------------------//


template <class T>
class SerialBenchGer : public SerialBench  {
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
      ger(dim, dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Ger, SerialBenchGer)


//----------------------- Syr Bench -----------------------------------------//


template <class T>
class SerialBenchSyr : public SerialBench  {
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
      syr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(SerialBench, Syr, SerialBenchSyr)


//----------------------- Syr2 Bench -----------------------------------------//


template <class T>
class SerialBenchSyr2 : public SerialBench  {
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
      syr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Syr2, SerialBenchSyr2)


//----------------------- Spr Bench -----------------------------------------//


template <class T>
class SerialBenchSpr : public SerialBench  {
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
      spr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(SerialBench, Spr, SerialBenchSpr)


//----------------------- Spr2 Bench -----------------------------------------//


template <class T>
class SerialBenchSpr2 : public SerialBench  {
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
      spr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Spr2, SerialBenchSpr2)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class SerialBenchGemv : public SerialBench  {
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
      gemv(dim, dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Gemv, SerialBenchGemv)


//----------------------- Gbmv Bench -----------------------------------------//


template <class T>
class SerialBenchGbmv : public SerialBench  {
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
      gbmv(dim, dim, band, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Gbmv, SerialBenchGbmv)


//----------------------- Symv Bench -----------------------------------------//


template <class T>
class SerialBenchSymv : public SerialBench  {
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
      symv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Symv, SerialBenchSymv)


//----------------------- Sbmv Bench -----------------------------------------//


template <class T>
class SerialBenchSbmv : public SerialBench  {
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
      sbmv('u', dim, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Sbmv, SerialBenchSbmv)


//----------------------- Spmv Bench -----------------------------------------//


template <class T>
class SerialBenchSpmv : public SerialBench  {
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
      spmv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    Interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(SerialBench, Spmv, SerialBenchSpmv)


//----------------------- Trmv Bench -----------------------------------------//


template <class T>
class SerialBenchTrmv : public SerialBench  {
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
      trmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x;
};
BENCH_F_F(SerialBench, Trmv, SerialBenchTrmv)


//----------------------- Tbmv Bench -----------------------------------------//


template <class T>
class SerialBenchTbmv : public SerialBench  {
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
      tbmv('u', dim, band, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    Interval<T> *A, *x;
};
BENCH_F_F(SerialBench, Tbmv, SerialBenchTbmv)


//----------------------- Tpmv Bench -----------------------------------------//


template <class T>
class SerialBenchTpmv : public SerialBench  {
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
      tpmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    Interval<T> *A, *x;
};
BENCH_F_F(SerialBench, Tpmv, SerialBenchTpmv)


//----------------------- Trsv Bench -----------------------------------------//


template <class T>
class SerialBenchTrsv : public SerialBench  {
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
      trsv('u', dim, A, x);
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
BENCH_F_F(SerialBench, Trsv, SerialBenchTrsv)


//----------------------- Tbsv Bench -----------------------------------------//


template <class T>
class SerialBenchTbsv : public SerialBench  {
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
      tbsv('u', dim, band, A, x);
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
BENCH_F_F(SerialBench, Tbsv, SerialBenchTbsv)


//----------------------- Tpsv Bench -----------------------------------------//


template <class T>
class SerialBenchTpsv : public SerialBench  {
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
      tpsv('u', dim, A, x);
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
BENCH_F_F(SerialBench, Tpsv, SerialBenchTpsv)


//----------------------- Gemm Bench -----------------------------------------//


template <class T>
class SerialBenchGemm : public SerialBench  {
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
      gemm(dim, dim, dim, alpha, beta, A, B, C);
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
BENCH_F_F(SerialBench, Gemm, SerialBenchGemm)


//----------------------- Symm Bench -----------------------------------------//


template <class T>
class SerialBenchSymm : public SerialBench  {
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
      symm('l', 'u', dim, dim, alpha, beta, A, B, C);
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
BENCH_F_F(SerialBench, Symm, SerialBenchSymm)


//----------------------- Syrk Bench -----------------------------------------//


template <class T>
class SerialBenchSyrk : public SerialBench  {
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
      syrk( 'u', dim, dim, alpha, beta, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, alpha, beta;
};
BENCH_F_F(SerialBench, Syrk, SerialBenchSyrk)


//----------------------- Syr2k Bench -----------------------------------------//


template <class T>
class SerialBenchSyr2k : public SerialBench  {
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
      syr2k('u', dim, dim, alpha, beta, A, B, C);
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
BENCH_F_F(SerialBench, Syr2k, SerialBenchSyr2k)


//----------------------- Trmm Bench -----------------------------------------//


template <class T>
class SerialBenchTrmm : public SerialBench  {
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
      trmm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    Interval<T> *A, *B, alpha;
};
BENCH_F_F(SerialBench, Trmm, SerialBenchTrmm)


//----------------------- Trsm Bench -----------------------------------------//


template <class T>
class SerialBenchTrsm : public SerialBench  {
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
      trsm('l', 'u', dim, dim, alpha, A, B);
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
BENCH_F_F(SerialBench, Trsm, SerialBenchTrsm)



} // namespace bench
} // namespace intlag

#endif



