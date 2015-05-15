
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BENCH_H
#define CUDA_BENCH_H

//#include <cmath>

#include "aux/case.h"
#include "aux/reference.h"

//#include "../include/blas/omp_blas.h"
#include "../include/blas/cuda_blas.h"

#include "gtest/gtest.h"
#include "aux/test_interval.h"



namespace intlag {
namespace bench {


//----------------------- Fixture --------------------------------------------//
class CudaBench : public BenchTest {
  public:

    CudaBench() {
      r = Reference::getInstance();

      int n = r->length;
      CudaInterval<double> alpha = r->alpha;
      CudaInterval<double> *x = new CudaInterval<double> [n];
      acopy(n, r->x, x);
      scal(n, alpha, x);
    }
    virtual ~CudaBench() {}

    void SetUp() {}

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
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      scal(n, alpha, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    CudaInterval<T> *x, alpha;
};
BENCH_FD_F(CudaBench, Scal, CudaBenchScal)


//----------------------- AXPY Bench -----------------------------------------//

template <class T>
class CudaBenchAXPY : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      alpha = r->alpha;
      x = new CudaInterval<T> [n];
      y = new CudaInterval<T> [n];
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
    CudaInterval<T> *x, *y, alpha;
};
BENCH_FD_F(CudaBench, AXPY, CudaBenchAXPY)


//----------------------- ASum Bench -----------------------------------------//

template <class T>
class CudaBenchASum : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      asum(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    CudaInterval<T> *x, ret;
};
BENCH_FD_F(CudaBench, Asum, CudaBenchASum)


//----------------------- Dot Bench -----------------------------------------//

template <class T>
class CudaBenchDot : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      y = new CudaInterval<T> [n];
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
    CudaInterval<T> *x, *y, ret;
};
BENCH_FD_F(CudaBench, Dot, CudaBenchDot)


//----------------------- Norm2 Bench -----------------------------------------//

template <class T>
class CudaBenchNorm2 : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      CudaGeneralManaged::norm2(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    CudaInterval<T> *x, ret;
};
BENCH_FD_F(CudaBench, Norm2, CudaBenchNorm2)


//----------------------- Rot Bench -----------------------------------------//

template <class T>
class CudaBenchRot : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      y = new CudaInterval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      CudaGeneralManaged::rot(n, x, y, 0.5, 0.5);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    CudaInterval<T> *x, *y;
};
BENCH_FD_F(CudaBench, Rot, CudaBenchRot)


//----------------------- Rotm Bench -----------------------------------------//

template <class T>
class CudaBenchRotm : public CudaBench  {
   public:

    void begin() {
      h[0] = -1; h[1] = 0.5; h[2] = 0.6; h[3] = 0.7; h[4] = 0.8;
      n = r->length;
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
      y = new CudaInterval<T> [n];
      acopy(n, r->y, y);
    }

    void run() {
      CudaGeneralManaged::rotm(n, x, y, h);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    T h[5];
    CudaInterval<T> *x, *y;
};
BENCH_FD_F(CudaBench, Rotm, CudaBenchRotm)


//----------------------- Ger Bench -----------------------------------------//


template <class T>
class CudaBenchGer : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      A = new CudaInterval<T> [n];
      x = new CudaInterval<T> [dim];
      y = new CudaInterval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->x, x);
      acopy(dim, r->y, y);
    }

    void run() {
      CudaGeneralManaged::ger(dim, dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_FD_F(CudaBench, Ger, CudaBenchGer)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class CudaBenchGemv : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new CudaInterval<T> [n];
      x = new CudaInterval<T> [dim];
      y = new CudaInterval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->y, x);
      acopy(dim, r->y, y);
    }

    void run() {
      //DeviceData< CudaInterval<T> > dA(n, A), dx(dim, x), dy(dim, y);
      //CudaGeneralManaged::gemv(dim, dim, alpha, beta, dA.data(), dx.data(), dy.data());
      CudaGeneralManaged::gemv(dim, dim, alpha, beta, A, x, y);
      //dy.toHost(y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_FD_F(CudaBench, Gemv, CudaBenchGemv)


//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class CudaBenchGemvShared : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new CudaInterval<T> [n];
      x = new CudaInterval<T> [dim];
      y = new CudaInterval<T> [dim];
      acopy(n, r->x, A);
      acopy(dim, r->y, x);
      acopy(dim, r->y, y);
    }

    void run() {
      //DeviceData< CudaInterval<T> > dA(n, A), dx(dim, x), dy(dim, y);
      //CudaGeneralManaged::gemv(dim, dim, alpha, beta, dA.data(), dx.data(), dy.data());
      CudaSharedManaged::gemv(dim, dim, alpha, beta, A, x, y);
      //dy.toHost(y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_FD_F(CudaBench, GemvShared, CudaBenchGemvShared)


//----------------------- Gemm Bench -----------------------------------------//

template <class T>
class CudaBenchGemm : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new CudaInterval<T> [n];
      B = new CudaInterval<T> [n];
      C = new CudaInterval<T> [n];
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      //DeviceData< CudaInterval<T> > dA(n, A);
      //DeviceData< CudaInterval<T> > dB(n, B);
      //DeviceData< CudaInterval<T> > dC(n, C);
      //gemm(dim, dim, dim, alpha, beta, dA.data(), dB.data(), dC.data());
      CudaGeneralManaged::gemm(dim, dim, dim, alpha, beta, A, B, C);
      //dC.toHost(C);
    }

    void check() {
      CudaInterval<T> *x, *y, *z;
      x = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      y = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      z = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      acopy(n, r->x, x);
      acopy(n, r->y, y);
      acopy(n, r->y, z);
      acopy(n, r->y, C);

      gemm(dim, dim, dim, alpha, beta, x, y, z);
      CudaGeneralManaged::gemm(dim, dim, dim, alpha, beta, A, B, C);

      for(int i = 0; i < n; ++i)
        EXPECT_DINTERVAL_NEAR(z[i], C[i], 0.1);

      free(x);
      free(y);
      free(z);
    }

    short iterations() { return 100;}

    void end() {
      delete[] A;
      delete[] B;
      delete[] C;
    }


    int n, dim;
    CudaInterval<T> *A, *B, *C, alpha, beta;
};
BENCH_FD_F(CudaBench, Gemm, CudaBenchGemm)


//----------------------- GemmShared Bench -----------------------------------------//


template <class T>
class CudaBenchGemmShared : public CudaBench  {
   public:

    void begin() {
      n = r->length;
      dim = std::sqrt(n);
      alpha = r->alpha;
      beta = r->beta;
      A = new CudaInterval<T> [n];
      B = new CudaInterval<T> [n];
      C = new CudaInterval<T> [n];
      acopy(n, r->x, A);
      acopy(n, r->y, B);
      acopy(n, r->y, C);
    }

    void run() {
      //DeviceData< CudaInterval<T> > dA(n, A);
      //DeviceData< CudaInterval<T> > dB(n, B);
      //DeviceData< CudaInterval<T> > dC(n, C);
      //gemm(dim, dim, dim, alpha, beta, dA.data(), dB.data(), dC.data());
      CudaSharedManaged::gemm(dim, dim, dim, alpha, beta, A, B, C);
      //dC.toHost(C);
    }

    void check() {
      CudaInterval<T> *x, *y, *z;
      x = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      y = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      z = (CudaInterval<T>*) malloc(n*sizeof(CudaInterval<T>));
      acopy(n, r->x, x);
      acopy(n, r->y, y);
      acopy(n, r->y, z);
      acopy(n, r->y, C);

      gemm(dim, dim, dim, alpha, beta, x, y, z);
      CudaSharedManaged::gemm(dim, dim, dim, alpha, beta, A, B, C);

      for(int i = 0; i < n; ++i)
        EXPECT_DINTERVAL_NEAR(z[i], C[i], 0.1);

      free(x);
      free(y);
      free(z);
    }

    short iterations() { return 100;}

    void end() {
      delete[] A;
      delete[] B;
      delete[] C;
    }


    int n, dim;
    CudaInterval<T> *A, *B, *C, alpha, beta;
};
BENCH_FD_F(CudaBench, GemmShared, CudaBenchGemmShared)


} // namespace bench
} // namespace intlag

#endif



