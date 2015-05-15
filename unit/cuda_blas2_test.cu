
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BLAS2_TEST
#define CUDA_BLAS2_TEST

#include "gtest/gtest.h"
#include "aux/reference.h"
#include "aux/test_interval.h"

#include "blas/serial_blas.h"
#include "blas/omp_blas.h"
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


TEST_F(CudaBlasTest, Ger) {

  CudaInterval<double> A1[10000], B1[10000], C1[10000];
  Interval<double> y1[10000], x1[10000], z1[10000];
  acopy(10000, ref->x, x1);
  acopy(10000, ref->y, y1);
  acopy(10000, ref->x, z1);
  acopy(10000, ref->x, A1);
  acopy(10000, ref->y, B1);
  acopy(10000, ref->x, C1);

  omp::ger(99, 98, 3.0, x1, y1, z1);
  CudaGeneralManaged::ger(99, 98, 3.0, A1, B1, C1);
  for(int i = 0; i < 99*98; ++i)
    EXPECT_DINTERVAL_EQ(z1[i], C1[i]);
}


TEST_F(CudaBlasTest, Syr) {

}


TEST_F(CudaBlasTest, Syr2) {

}


TEST_F(CudaBlasTest, Spr) {

}


TEST_F(CudaBlasTest, Spr2) {

}


TEST_F(CudaBlasTest, Gemv) {

  CudaInterval<double> A1[10000], B1[10000], C1[10000];
  Interval<double> y1[10000], x1[10000], z1[10000];
  acopy(10000, ref->x, x1);
  acopy(10000, ref->y, y1);
  acopy(10000, ref->x, z1);
  acopy(10000, ref->x, A1);
  acopy(10000, ref->y, B1);
  acopy(10000, ref->x, C1);

  gemv(99, 69, -33.33, 22.24, x1, y1, z1);
  CudaGeneralManaged::gemv(99, 69, -33.33, 22.24, A1, B1, C1);
  for(int i = 0; i < 99; ++i)
    EXPECT_DINTERVAL_EQ(z1[i], C1[i]);
}

TEST_F(CudaBlasTest, GemvShared) {

  CudaInterval<double> A1[10000], B1[10000], C1[10000];
  Interval<double> y1[10000], x1[10000], z1[10000];
  acopy(10000, ref->x, x1);
  acopy(10000, ref->y, y1);
  acopy(10000, ref->x, z1);
  acopy(10000, ref->x, A1);
  acopy(10000, ref->y, B1);
  acopy(10000, ref->x, C1);

  gemv(99, 98, -33.33, 22.24, x1, y1, z1);
  CudaSharedManaged::gemv(99, 98, -33.33, 22.24, A1, B1, C1);
  for(int i = 0; i < 99; ++i) {
    EXPECT_DINTERVAL_EQ(z1[i], C1[i]);
  }
}


TEST_F(CudaBlasTest, Gbmv) {

}


TEST_F(CudaBlasTest, Symv) {

}


TEST_F(CudaBlasTest, Sbmv) {

}


TEST_F(CudaBlasTest, Spmv) {

}


TEST_F(CudaBlasTest, Trmv) {

}


TEST_F(CudaBlasTest, Tbmv) {

}


TEST_F(CudaBlasTest, Tpmv) {

}


TEST_F(CudaBlasTest, Trsv) {

}


TEST_F(CudaBlasTest, Tbsv) {

}


TEST_F(CudaBlasTest, Tpsv) {

}


#endif


