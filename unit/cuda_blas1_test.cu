
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BLAS1_TEST
#define CUDA_BLAS1_TEST

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


TEST_F(CudaBlasTest, ACopy) {

	acopy(3, x, z);

	EXPECT_DOUBLE_EQ(1.1, z[0].inf());
	EXPECT_DOUBLE_EQ(3.3, z[1].inf());
	EXPECT_DOUBLE_EQ(5.5, z[2].inf());
	EXPECT_DOUBLE_EQ(2.2, z[0].sup());
	EXPECT_DOUBLE_EQ(4.4, z[1].sup());
	EXPECT_DOUBLE_EQ(6.6, z[2].sup());
}


TEST_F(CudaBlasTest, Swap) {

	swap(3, x, y);

	EXPECT_DOUBLE_EQ(1.1, y[0].inf());
	EXPECT_DOUBLE_EQ(3.3, y[1].inf());
	EXPECT_DOUBLE_EQ(5.5, y[2].inf());
	EXPECT_DOUBLE_EQ(2.2, y[0].sup());
	EXPECT_DOUBLE_EQ(4.4, y[1].sup());
	EXPECT_DOUBLE_EQ(6.6, y[2].sup());

	EXPECT_DOUBLE_EQ(10.1, x[0].inf());
	EXPECT_DOUBLE_EQ(30.2, x[1].inf());
	EXPECT_DOUBLE_EQ(-500.0, x[2].inf());
	EXPECT_DOUBLE_EQ(20.1, x[0].sup());
	EXPECT_DOUBLE_EQ(40.2, x[1].sup());
	EXPECT_DOUBLE_EQ(-300.0, x[2].sup());
}


TEST_F(CudaBlasTest, Scal) {

	CudaGeneralManaged::scal(3, (double)2, x);

	EXPECT_DOUBLE_EQ(2*1.1, x[0].inf());
	EXPECT_DOUBLE_EQ(2*3.3, x[1].inf());
	EXPECT_DOUBLE_EQ(2*5.5, x[2].inf());
	EXPECT_DOUBLE_EQ(2*2.2, x[0].sup());
	EXPECT_DOUBLE_EQ(2*4.4, x[1].sup());
	EXPECT_DOUBLE_EQ(2*6.6, x[2].sup());
}


TEST_F(CudaBlasTest, AXPY) {

	CudaGeneralManaged::axpy(3, 2, x, y);

	EXPECT_DOUBLE_EQ(2*1.1+10.1, y[0].inf());
	EXPECT_DOUBLE_EQ(2*3.3+30.2, y[1].inf());
	EXPECT_DOUBLE_EQ(2*5.5-500, y[2].inf());
	EXPECT_DOUBLE_EQ(2*2.2+20.1, y[0].sup());
	EXPECT_DOUBLE_EQ(2*4.4+40.2, y[1].sup());
	EXPECT_DOUBLE_EQ(2*6.6-300, y[2].sup());
}


TEST_F(CudaBlasTest, ASum) {

	CudaGeneralManaged::asum(3, &r, x);

	EXPECT_DOUBLE_EQ(9.9, r.inf());
	EXPECT_DOUBLE_EQ(13.2, r.sup());
}


TEST_F(CudaBlasTest, Dot) {

	CudaGeneralManaged::dot(3, &r, x, y);

	//EXPECT_DOUBLE_EQ(0, r.inf());
	EXPECT_DOUBLE_EQ(2.2*20.1 + 4.4*40.2 + 500*6.6, r.sup());
}


TEST_F(CudaBlasTest, Norm2) {

	Interval<double> x1[99], r1;
	CudaInterval<double> x2[99];
	acopy(99, ref->x, x1);
	acopy(99, ref->x, x2);

	norm2(99, &r1, x1);
	CudaGeneralManaged::norm2(99, &r, x2);

	// SQRT is not exact
	EXPECT_NEAR(r1.inf(), r.inf(), 1e-6);
	EXPECT_NEAR(r1.sup(), r.sup(), 1e-6);
}


TEST_F(CudaBlasTest, Rot) {

  Interval<double> x1[34], y1[34];
  CudaInterval<double> x2[34], y2[34];
  acopy(34, ref->x, x1);
  acopy(34, ref->y, y1);
  acopy(34, ref->x, x2);
  acopy(34, ref->y, y2);

  omp::rot(34, x1, y1, 0.5, 0.6);
  CudaGeneralManaged::rot(34, x2, y2, 0.5, 0.6);
  for(int i = 0; i < 34; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


TEST_F(CudaBlasTest, Rotm) {

  Interval<double> x1[34], y1[34];
  CudaInterval<double> x2[34], y2[34];
  acopy(34, ref->x, x1);
  acopy(34, ref->y, y1);
  acopy(34, ref->x, x2);
  acopy(34, ref->y, y2);

  double H2[] = {-2,15,16,17,-18};
  omp::rotm(34, x1, y1, H2);
  CudaGeneralManaged::rotm(34, x2, y2, H2);
  for(int i = 0; i < 34; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

  double H1[] = {1,0,0,0,0};
  omp::rotm(34, x1, y1, H1);
  CudaGeneralManaged::rotm(34, x2, y2, H1);
  for(int i = 0; i < 34; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

  double H0[] = {0,0,0,0,0};
  omp::rotm(34, x1, y1, H0);
  CudaGeneralManaged::rotm(34, x2, y2, H0);
  for(int i = 0; i < 34; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

  double H[] = {-1,0,0,0,0};
  omp::rotm(34, x1, y1, H);
  CudaGeneralManaged::rotm(34, x2, y2, H);
  for(int i = 0; i < 34; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


#endif



