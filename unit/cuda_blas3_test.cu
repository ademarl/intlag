
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BLAS3_TEST
#define CUDA_BLAS3_TEST

#include "gtest/gtest.h"
#include "aux/reference.h"
#include "aux/test_interval.h"

#include "blas/serial_blas.h"
#include "blas/cuda_blas.h"

using namespace intlag;


class CudaBlasTest : public ::testing::Test{

	protected:

    CudaBlasTest () {
      ref = Reference::getInstance();
    }

	  virtual void SetUp() {

		  x[0] = CudaInterval<double>(1.1, 2.2);
		  x[1] = CudaInterval<double>(3.3, 4.4);
		  x[2] = CudaInterval<double>(5.5, 6.6);

		  y[0] = CudaInterval<double>(10.1, 20.1);
		  y[1] = CudaInterval<double>(30.2, 40.2);
		  y[2] = CudaInterval<double>(-500.0, -300.0);

	    A[0] = CudaInterval<double>(1.0);
      A[1] = CudaInterval<double>(1.0);
      A[2] = CudaInterval<double>(2.0);
      A[3] = CudaInterval<double>(4.0);
	    B[0] = CudaInterval<double>(1.0, 2.0);
      B[1] = CudaInterval<double>(3.0, 4.0);
	    C[0] = CudaInterval<double>(1.0, 2.0);
      C[1] = CudaInterval<double>(3.0, 4.0);
	  }

	  virtual void TearDown() { }

	  CudaInterval<double> x[3], y[3], z[3], A[4], B[3], C[3], r;
    Reference* ref;
};


TEST_F(CudaBlasTest, Gemm) {

  CudaInterval<double> A[10000], B[10000], C[10000], D[10000];
  Interval<double> y[10000], x[10000], z[10000];
  acopy(10000, ref->x, x);
  acopy(10000, ref->y, y);
  acopy(10000, ref->x, z);
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, B);
  acopy(10000, ref->x, C);

  gemm(99, 98, 76, -33.33, 22.24, x, y, z);
  CudaGeneralManaged::gemm(99, 98, 76, -33.33, 22.24, A, B, C);
  for(int i = 0; i < 99*98; ++i)
    EXPECT_DINTERVAL_EQ(z[i], C[i]);
}


TEST_F(CudaBlasTest, GemmShared) {

  CudaInterval<double> A[10000], B[10000], C[10000], D[10000];
  Interval<double> y[10000], x[10000], z[10000];
  acopy(10000, ref->x, x);
  acopy(10000, ref->y, y);
  acopy(10000, ref->x, z);
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, B);
  acopy(10000, ref->x, C);

  gemm(99, 98, 76, -33.33, 22.24, x, y, z);
  CudaSharedManaged::gemm(99, 98, 76, -33.33, 22.24, A, B, C);
  for(int i = 0; i < 99*98; ++i)
    EXPECT_DINTERVAL_EQ(z[i], C[i]);
}


TEST_F(CudaBlasTest, Symm) {

}


TEST_F(CudaBlasTest, Syrk) {

}


TEST_F(CudaBlasTest, Syrk2) {

}


TEST_F(CudaBlasTest, Trmm) {

}


TEST_F(CudaBlasTest, Trsm) {

}


#endif



