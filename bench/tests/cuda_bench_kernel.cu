
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BENCH_KERNEL_H
#define CUDA_BENCH_KERNEL_H

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
class CudaKernelBench : public BenchTest {
  public:

    CudaKernelBench() {
      r = Reference::getInstance();
    }
    virtual ~CudaKernelBench() {}

    void SetUp() {
      int n = r->length;
      CudaInterval<double> alpha = r->alpha;
      CudaInterval<double> *x = new CudaInterval<double> [n];
      acopy(n, r->x, x);
      scal(n, alpha, x);
    }

    void TearDown() {}

    Reference* r;
};


//----------------------- Scal Bench -----------------------------------------//

template <class T>
class CudaKernelBenchScal : public CudaKernelBench  {
   public:

    void begin() {
      n = r->length;
      alpha = r->alpha;
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      DeviceData< CudaInterval<T> > dx(n,x);
	    //CudaGeneral::scal(n, alpha, dx.data());
      dx.toHost(x);
    }

    void end() {
      delete[] x;
    }

    int n;
    CudaInterval<T> *x, alpha;
};
//BENCH_FD_F(CudaKernelBench, Scal, CudaKernelBenchScal)


//----------------------- AXPY Bench -----------------------------------------//

template <class T>
class CudaKernelBenchAXPY : public CudaKernelBench  {
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
      DeviceData< CudaInterval<T> > dx(n,x), dy(n,y);
      //CudaGeneral::axpy(n, alpha, dx.data(), dy.data());
      dy.toHost(y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    CudaInterval<T> *x, *y, alpha;
};
BENCH_FD_F(CudaKernelBench, AXPY, CudaKernelBenchAXPY)

/*
//----------------------- ASum Bench -----------------------------------------//

template <class T>
class CudaKernelBenchASum : public CudaKernelBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      CudaGeneral::asum(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    CudaInterval<T> *x, ret;
};
BENCH_F_F(CudaKernelBench, Asum, CudaKernelBenchASum)


//----------------------- Dot Bench -----------------------------------------//

template <class T>
class CudaKernelBenchDot : public CudaKernelBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      y = new CudaInterval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      CudaGeneral::dot(n, &ret, x, y);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    CudaInterval<T> *x, *y, ret;
};
BENCH_F_F(CudaKernelBench, Dot, CudaKernelBenchDot)


//----------------------- Norm2 Bench -----------------------------------------//

template <class T>
class CudaKernelBenchNorm2 : public CudaKernelBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      acopy(n, r->x, x);
    }

    void run() {
      CudaGeneral::norm2(n, &ret, x);
    }

    void end() {
      delete[] x;
    }

    int n;
    CudaInterval<T> *x, ret;
};
BENCH_F_F(CudaKernelBench, Norm2, CudaKernelBenchNorm2)

//----------------------- Rot Bench -----------------------------------------//

template <class T>
class CudaKernelBenchRot : public CudaKernelBench  {
   public:

    void begin() {
      n = r->length;
      x = new CudaInterval<T> [n];
      y = new CudaInterval<T> [n];
      acopy(n, r->x, x);
      acopy(n, r->y, y);
    }

    void run() {
      CudaGeneral::rot(n, x, y, 0.5, 0.5);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    CudaInterval<T> *x, *y;
};
BENCH_F_F(CudaKernelBench, Rot, CudaKernelBenchRot)


//----------------------- Rotm Bench -----------------------------------------//

template <class T>
class CudaKernelBenchRotm : public CudaKernelBench  {
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
      CudaGeneral::rotm(n, x, y, h);
    }

    void end() {
      delete[] x;
      delete[] y;
    }

    int n;
    T h[5];
    CudaInterval<T> *x, *y;
};
BENCH_F_F(CudaKernelBench, Rotm, CudaKernelBenchRotm)


//----------------------- Ger Bench -----------------------------------------//


template <class T>
class CudaKernelBenchGer : public CudaKernelBench  {
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
      CudaGeneral::ger(dim, dim, alpha, x, y, A);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_F_F(CudaKernelBench, Ger, CudaKernelBenchGer)
*/

//----------------------- Gemv Bench -----------------------------------------//


template <class T>
class CudaKernelBenchGemv : public CudaKernelBench  {
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
      DeviceData< CudaInterval<T> > dA(n, A), dx(dim, x), dy(dim, y);
      //CudaGeneralManaged::gemv(dim, dim, alpha, beta, dA.data(), dx.data(), dy.data());
      //CudaGeneral::gemv(dim, dim, alpha, beta, A, x, y);
      dy.toHost(y);
    }

    void end() {
      delete[] A;
      delete[] x;
      delete[] y;
    }

    int n, dim;
    CudaInterval<T> *A, *x, *y, alpha, beta;
};
BENCH_FD_F(CudaKernelBench, Gemv, CudaKernelBenchGemv)


//----------------------- Gemv Bench -----------------------------------------//

/*
template <class T>
class CudaKernelBenchGemvShared : public CudaKernelBench  {
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
      CudaShared::gemv(dim, dim, alpha, beta, A, x, y);
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
BENCH_FD_F(CudaKernelBench, GemvShared, CudaKernelBenchGemvShared)

*/
//----------------------- Gemm Bench -----------------------------------------//

template <class T>
class CudaKernelBenchGemm : public CudaKernelBench  {
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
      DeviceData< CudaInterval<T> > dA(n, A);
      DeviceData< CudaInterval<T> > dB(n, B);
      DeviceData< CudaInterval<T> > dC(n, C);
      //gemm(dim, dim, dim, alpha, beta, dA.data(), dB.data(), dC.data());
      //CudaGeneral::gemm(dim, dim, dim, alpha, beta, dx.data(), dy.data(), dz.data());
      dC.toHost(C);
    }

    void check() {
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
BENCH_FD_F(CudaKernelBench, Gemm, CudaKernelBenchGemm)
/*

//----------------------- GemmShared Bench -----------------------------------------//


template <class T>
class CudaKernelBenchGemmShared : public CudaKernelBench  {
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
      CudaShared::gemm(dim, dim, dim, alpha, beta, A, B, C);
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
      CudaShared::gemm(dim, dim, dim, alpha, beta, A, B, C);

      for(int i = 0; i < n; ++i)
        EXPECT_DINTERVAL_NEAR(z[i], C[i], 0.1);

      free(x);
      free(y);
      free(z);
    }

    short iterations() { return 1;}

    void end() {
      delete[] A;
      delete[] B;
      delete[] C;
    }


    int n, dim;
    CudaInterval<T> *A, *B, *C, alpha, beta;
};
BENCH_FD_F(CudaKernelBench, GemmShared, CudaKernelBenchGemmShared)*/


} // namespace bench
} // namespace intlag

#endif



