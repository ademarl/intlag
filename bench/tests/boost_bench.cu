
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef BOOST_BENCH_H
#define BOOST_BENCH_H

#include "boost/numeric/interval.hpp"
using boost::numeric::interval;
using namespace boost::numeric;


#include "aux/case.h"
#include "aux/reference.h"
#include "../include/blas/boost_serial_blas.h"

#include "gtest/gtest.h"



namespace intlag {
namespace bench {

template<class T, class U>
void acopy(int n, intlag::Interval<T> const *x, interval<U> *y) {
		for(int i = 0; i < n; ++i)
			y[i] = interval<T>(x[i].inf(), x[i].sup());
}

//----------------------- Fixture --------------------------------------------//

class BoostBench : public BenchTest {
  public:

    BoostBench() {
      r = Reference::getInstance();
    }
    virtual ~BoostBench() {}

    void SetUp() {}
    void TearDown() {}

    Reference* r;
};


//----------------------- Scal Bench -----------------------------------------//

template <class T>
class BoostBenchScal : public BoostBench  {
   public:

    void begin() {


      n = r->length;
      alpha = interval<T>((r->alpha).inf(), (r->alpha).sup());
      x = new interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      bi::scal(n, alpha, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    interval<T> *x, alpha;
};
BENCH_F_F(BoostBench, Scal, BoostBenchScal)


//----------------------- AXPY Bench -----------------------------------------//

template <class T>
class BoostBenchAXPY : public BoostBench  {
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
      bi::axpy(n, alpha, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    interval<T> *x, *y, alpha;
};
BENCH_F_F(BoostBench, AXPY, BoostBenchAXPY)


//----------------------- ASum Bench -----------------------------------------//

template <class T>
class BoostBenchASum : public BoostBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      bi::asum(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    interval<T> *x, ret;
};
BENCH_F_F(BoostBench, Asum, BoostBenchASum)


//----------------------- Dot Bench -----------------------------------------//

template <class T>
class BoostBenchDot : public BoostBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      y = new interval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      bi::dot(n, &ret, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    interval<T> *x, *y, ret;
};
BENCH_F_F(BoostBench, Dot, BoostBenchDot)


//----------------------- Norm2 Bench -----------------------------------------//

template <class T>
class BoostBenchNorm2 : public BoostBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      bi::norm2(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    interval<T> *x, ret;
};
BENCH_F_F(BoostBench, Norm2, BoostBenchNorm2)


//----------------------- Rot Bench -----------------------------------------//

template <class T>
class BoostBenchRot : public BoostBench  {
   public:

    void begin() {
      n = r->length;
      x = new interval<T> [n];
      acopy(n, r->x, x);
      y = new interval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      bi::rot(n, x, y, interval<T>(0.5), interval<T>(0.5));
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    interval<T> *x, *y;
};
BENCH_F_F(BoostBench, Rot, BoostBenchRot)


//----------------------- Rotm Bench -----------------------------------------//

template <class T>
class BoostBenchRotm : public BoostBench  {
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
      bi::rotm(n, x, y, h);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    T h[5];
    interval<T> *x, *y;
};
BENCH_F_F(BoostBench, Rotm, BoostBenchRotm)


//----------------------- Ger Bench -----------------------------------------//


template <class T>
class BoostBenchGer : public BoostBench  {
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
      bi::ger(dim, dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Ger, BoostBenchGer)


//----------------------- Syr Bench -----------------------------------------//


template <class T>
class BoostBenchSyr : public BoostBench  {
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
      bi::syr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(BoostBench, Syr, BoostBenchSyr)


//----------------------- Syr2 Bench -----------------------------------------//


template <class T>
class BoostBenchSyr2 : public BoostBench  {
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
      bi::syr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Syr2, BoostBenchSyr2)


//----------------------- Spr Bench -----------------------------------------//


template <class T>
class BoostBenchSpr : public BoostBench  {
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
      bi::spr('u', dim, alpha, x, A);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x, alpha, beta;
};
BENCH_F_F(BoostBench, Spr, BoostBenchSpr)


//----------------------- Spr2 Bench -----------------------------------------//


template <class T>
class BoostBenchSpr2 : public BoostBench  {
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
      bi::spr2('u', dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Spr2, BoostBenchSpr2)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class BoostBenchGemv : public BoostBench  {
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
      bi::gemv(dim, dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Gemv, BoostBenchGemv)


//----------------------- Gbmv Bench -----------------------------------------//


template <class T>
class BoostBenchGbmv : public BoostBench  {
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
      bi::gbmv(dim, dim, band, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Gbmv, BoostBenchGbmv)


//----------------------- Symv Bench -----------------------------------------//


template <class T>
class BoostBenchSymv : public BoostBench  {
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
      bi::symv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Symv, BoostBenchSymv)


//----------------------- Sbmv Bench -----------------------------------------//


template <class T>
class BoostBenchSbmv : public BoostBench  {
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
      bi::sbmv('u', dim, band, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim, band;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Sbmv, BoostBenchSbmv)


//----------------------- Spmv Bench -----------------------------------------//


template <class T>
class BoostBenchSpmv : public BoostBench  {
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
      bi::spmv('u', dim, alpha, beta, A, x, y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    interval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(BoostBench, Spmv, BoostBenchSpmv)


//----------------------- Trmv Bench -----------------------------------------//


template <class T>
class BoostBenchTrmv : public BoostBench  {
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
      bi::trmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostBench, Trmv, BoostBenchTrmv)


//----------------------- Tbmv Bench -----------------------------------------//


template <class T>
class BoostBenchTbmv : public BoostBench  {
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
      bi::tbmv('u', dim, band, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    interval<T> *A, *x;
};
BENCH_F_F(BoostBench, Tbmv, BoostBenchTbmv)


//----------------------- Tpmv Bench -----------------------------------------//


template <class T>
class BoostBenchTpmv : public BoostBench  {
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
      bi::tpmv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostBench, Tpmv, BoostBenchTpmv)


//----------------------- Trsv Bench -----------------------------------------//


template <class T>
class BoostBenchTrsv : public BoostBench  {
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
      bi::trsv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostBench, Trsv, BoostBenchTrsv)


//----------------------- Tbsv Bench -----------------------------------------//


template <class T>
class BoostBenchTbsv : public BoostBench  {
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
      bi::tbsv('u', dim, band, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim, band;
    interval<T> *A, *x;
};
BENCH_F_F(BoostBench, Tbsv, BoostBenchTbsv)


//----------------------- Tpsv Bench -----------------------------------------//


template <class T>
class BoostBenchTpsv : public BoostBench  {
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
      bi::tpsv('u', dim, A, x);
    }

    void end() {
      delete[] A;
      delete[] x;
    }

    int n, dim;
    interval<T> *A, *x;
};
BENCH_F_F(BoostBench, Tpsv, BoostBenchTpsv)


//----------------------- Gemm Bench -----------------------------------------//


template <class T>
class BoostBenchGemm : public BoostBench  {
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
      bi::gemm(dim, dim, dim, alpha, beta, A, B, C);
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
BENCH_F_F(BoostBench, Gemm, BoostBenchGemm)


//----------------------- Symm Bench -----------------------------------------//


template <class T>
class BoostBenchSymm : public BoostBench  {
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
      bi::symm('l', 'u', dim, dim, alpha, beta, A, B, C);
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
BENCH_F_F(BoostBench, Symm, BoostBenchSymm)


//----------------------- Syrk Bench -----------------------------------------//


template <class T>
class BoostBenchSyrk : public BoostBench  {
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
      bi::syrk( 'u', dim, dim, alpha, beta, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, alpha, beta;
};
BENCH_F_F(BoostBench, Syrk, BoostBenchSyrk)


//----------------------- Syr2k Bench -----------------------------------------//


template <class T>
class BoostBenchSyr2k : public BoostBench  {
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
      bi::syr2k('u', dim, dim, alpha, beta, A, B, C);
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
BENCH_F_F(BoostBench, Syr2k, BoostBenchSyr2k)


//----------------------- Trmm Bench -----------------------------------------//


template <class T>
class BoostBenchTrmm : public BoostBench  {
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
      bi::trmm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, alpha;
};
BENCH_F_F(BoostBench, Trmm, BoostBenchTrmm)


//----------------------- Trsm Bench -----------------------------------------//


template <class T>
class BoostBenchTrsm : public BoostBench  {
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
      bi::trsm('l', 'u', dim, dim, alpha, A, B);
    }

    void end() {
      free(A);
      free(B);
    }

    short iterations() { return 1;}

    int n, dim;
    interval<T> *A, *B, alpha;
};
BENCH_F_F(BoostBench, Trsm, BoostBenchTrsm)



} // namespace bench
} // namespace intlag

#endif



