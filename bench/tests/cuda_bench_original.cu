
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BENCH_H
#define CUDA_BENCH_H

#include <cmath>

#include "aux/case.h"
#include "aux/reference.h"
#include "../include/blas/serial_blas.h"
#include "../include/blas/cuda_blas.h"
#include "gtest/gtest.h"



//FIXME: Separate References by template, think of it as a fixture



namespace intlag {
namespace bench {

//TODO: Compartimentalize this functions
template <class T>
void EXPECT_INTERVAL_EQ(CudaInterval<T> x, CudaInterval<T> y) {};

template<>
void EXPECT_INTERVAL_EQ(CudaInterval<float> x, CudaInterval<float> y) {
  EXPECT_FLOAT_EQ(x.inf(), y.inf());
  EXPECT_FLOAT_EQ(x.sup(), y.sup());
}

template<>
void EXPECT_INTERVAL_EQ(CudaInterval<double> x, CudaInterval<double> y) {
  EXPECT_DOUBLE_EQ(x.inf(), y.inf());
  EXPECT_DOUBLE_EQ(x.sup(), y.sup());
}


//----------------------- Fixture --------------------------------------------//
class CudaBench : public BenchTest {
  public:

    CudaBench() {
      r = Reference::getInstance();
    }
    virtual ~CudaBench() {}

    void SetUp() {
    }
    void TearDown() {}

    Reference* r;
};


//----------------------- Scal Bench -----------------------------------------//


template <class T>
class CudaBenchScal : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      alpha = r->alpha;
      x = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      y = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      acopy(n, r->x, x);
    }

    void run() {
      DeviceData< CudaInterval<T> > dx(n, x);
      CudaGeneral::scal(n, alpha, dx.data());
      dx.toHost(y);
    }

    void end() {
      free(x);
      free(y);
    }

    int n;
    CudaInterval<T> *x, *y, alpha;
};
BENCH_FD_F(CudaBench, Scal, CudaBenchScal)


//----------------------- AXPY Bench -----------------------------------------//


template <class T>
class CudaBenchAXPY : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      alpha = r->alpha;
      x = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      y = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      DeviceData< CudaInterval<T> > dx(n, x);
      DeviceData< CudaInterval<T> > dy(n, y);
      CudaGeneral::axpy(n, alpha, dx.data(), dy.data());
      dy.toHost(y);
    }

    void end() {
      free(x);
      free(y);
    }

    int n;
    CudaInterval<T> *x, *y, alpha;
};
BENCH_FD_F(CudaBench, AXPY, CudaBenchAXPY)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class CudaBenchGemv : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      x = (CudaInterval<T>*) malloc(dim*sizeof(CudaInterval<T>));
      y = (CudaInterval<T>*) malloc(dim*sizeof(CudaInterval<T>));
      acopy(n, r->x, A);
      acopy(dim, r->y, x);
      acopy(dim, r->y, y);
    }

    void run() {
      DeviceData< CudaInterval<T> > dA(n, A);
      DeviceData< CudaInterval<T> > dx(dim, x);
      DeviceData< CudaInterval<T> > dy(dim, y);
      CudaGeneral::gemv(dim, dim, alpha, beta, dA.data(), dx.data(), dy.data());
      dy.toHost(y);
    }

    void end() {
      free(A);
      free(x);
      free(y);
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_FD_F(CudaBench, Gemv, CudaBenchGemv)


//----------------------- Gemv Trans Bench -----------------------------------//


template <class T>
class CudaBenchGemvTrans : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      x = (CudaInterval<T>*) malloc(dim*sizeof(CudaInterval<T>));
      y = (CudaInterval<T>*) malloc(dim*sizeof(CudaInterval<T>));
      acopy(n, r->x, A);
      acopy(dim, r->y, x);
      acopy(dim, r->y, y);
    }

    void run() {
      CudaTrans::gemv(dim, dim, alpha, beta, A, x, y);
    }

    void end() {
      free(A);
      free(x);
      free(y);
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_FD_F(CudaBench, GemvTrans, CudaBenchGemvTrans)


//----------------------- Gemm Bench -----------------------------------------//


template <class T>
class CudaBenchGemm : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      B = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      C = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      DeviceData< CudaInterval<T> > dA(n, A);
      DeviceData< CudaInterval<T> > dB(n, B);
      DeviceData< CudaInterval<T> > dC(n, C);
      CudaGeneral::gemm(dim, dim, dim, alpha, beta, dA.data(), dB.data(), dC.data());
      dC.toHost(C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 10;}

    int n, dim;
    CudaInterval<T> *A, *B, *C, alpha, beta;
};
BENCH_FD_F(CudaBench, Gemm, CudaBenchGemm)


//----------------------- Gemm Trans Bench -----------------------------------//


template <class T>
class CudaBenchGemmTrans : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      B = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      C = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      CudaTrans::gemm(dim, dim, dim, alpha, beta, A, B, C);
    }

    void end() {
      free(A);
      free(B);
      free(C);
    }

    short iterations() { return 10;}

    int n, dim;
    CudaInterval<T> *A, *B, *C, alpha, beta;
};
BENCH_FD_F(CudaBench, GemmTrans, CudaBenchGemmTrans)



} // namespace bench
} // namespace intlag

#endif



