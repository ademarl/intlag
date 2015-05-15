
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef OMP_BLAS3_TEST_CU
#define OMP_BLAS3_TEST_CU

#include <math.h>
#include "blas/serial_blas.h"
#include "blas/omp_blas.h"
#include "aux/test_interval.h"
#include "aux/reference.h"
#include "gtest/gtest.h"

#include "aux/ulp.h"


using namespace intlag;


class OMPBlasTest : public ::testing::Test {
  protected:

    OMPBlasTest() {
      ref = Reference::getInstance();
    }
    virtual ~OMPBlasTest(){}

    virtual void SetUp()
    {
      for(int i = 0; i < 9; ++i) {
          a[i] = Interval<float>(i);
          b[i] = Interval<float>(-i, i);
          c[i] = Interval<float>(i-5, i+i*i);
      }
      for(int i = 0; i < 10000; ++i) {
          x[i] = Interval<float>(i);
          y[i] = Interval<float>(-i, i);
          z[i] = Interval<float>(i-5000, i+i*i);
      }

      for(int i = 0; i < 9; ++i) {
          da[i] = Interval<double>(i);
          db[i] = Interval<double>(-i, i);
          dc[i] = Interval<double>(i-5, i+i*i);
      }
      for(int i = 0; i < 10000; ++i) {
          dx[i] = Interval<double>(i);
          dy[i] = Interval<double>(-i, i);
          dz[i] = Interval<double>(i-5000, i+i*i);
      }
    }
    virtual void TearDown() {}

    Reference* ref;

    Interval<float> a[9], b[9], c[9];
    Interval<float> x[10000], y[10000], z[10000];
    Interval<double> da[9], db[9], dc[9], dx[10000], dy[10000], dz[10000];
};




/******************************** Tests ***************************************/


TEST_F(OMPBlasTest, Gemm) {

  Interval<float> out[4];

  acopy(4, b, out);
  omp::gemm(1, 1, 1, -2.5, 3.5, c, a, out);
  EXPECT_FINTERVAL_EQ(out[0], -2.5*c[0]*a[0]+3.5*b[0]);

  acopy(4, b, out);
  omp::gemm(2, 2, 1, -1.0, -5.5, c, b, out);
  EXPECT_FINTERVAL_EQ(out[0], Interval<float>(0.0));
  EXPECT_FINTERVAL_EQ(out[1], Interval<float>(-10.5,10.5));
  EXPECT_FINTERVAL_EQ(out[2], Interval<float>(-11.0,11.0));
  EXPECT_FINTERVAL_EQ(out[3], Interval<float>(-20.5,20.5));

  omp::gemm(3, 1, 3, 2.0, 3.0, c, a, b);
  EXPECT_FINTERVAL_EQ(b[0], Interval<float>(-20.0, 28.0));
  EXPECT_FINTERVAL_EQ(b[1], Interval<float>(-5.0, 163.0));
  EXPECT_FINTERVAL_EQ(b[2], Interval<float>(10.0, 406.0));

  Interval<double> x[10000], y[10000], z[10000], w[10000];
  acopy(10000, ref->x, x);
  acopy(10000, ref->y, y);
  acopy(10000, ref->x, z);
  acopy(10000, ref->x, w);

  gemm(340, 13, 27, 4.0, -3.5, x, y, w);
  omp::gemm(340, 13, 27, 4.0, -3.5, x, y, z);

  for(int i = 0; i < 340*13; ++i)
    EXPECT_FINTERVAL_EQ(z[i], w[i]);

};


TEST_F(OMPBlasTest, Symm) {

  Interval<double> A[10000], B[10000], C2[10000], C1[10000];
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, B);
  acopy(10000, ref->x, C1);
  acopy(10000, ref->x, C2);


  symm('l', 'u', 20, 129, -4.0, 3.5, A, B, C1);
  omp::symm('l', 'u', 20, 129, -4.0, 3.5, A, B, C2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);

  symm('l', 'l', 20, 129, -4.0, 3.5, A, B, C1);
  omp::symm('l', 'l', 20, 129, -4.0, 3.5, A, B, C2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);

  symm('r', 'U', 20, 129, -4.0, 3.5, A, B, C1);
  omp::symm('r', 'U', 20, 129, -4.0, 3.5, A, B, C2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);

  symm('r', 'l', 20, 129, -4.0, 3.5, A, B, C1);
  omp::symm('r', 'l', 20, 129, -4.0, 3.5, A, B, C2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);
}


TEST_F(OMPBlasTest, Syrk) {

  Interval<double> A[10000], C2[10000], C1[10000];
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, C1);
  acopy(10000, ref->y, C2);


  syrk('u', 21, 128, -4.0, 3.5, A, C1);
  omp::syrk('u', 21, 128, -4.0, 3.5, A, C2);
  for(int i = 0; i < 21*128; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);

  syrk('l', 20, 129, -4.0, 3.5, A, C1);
  omp::syrk('l', 20, 129, -4.0, 3.5, A, C2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);
}


TEST_F(OMPBlasTest, Syrk2) {

  Interval<double> A[10000], B[10000], C2[10000], C1[10000];
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, B);
  acopy(10000, ref->x, C1);
  acopy(10000, ref->x, C2);


  syr2k('u', 21, 128, -4.0, 3.5, A, B, C1);
  omp::syr2k('u', 21, 128, -4.0, 3.5, A, B, C2);
  for(int i = 0; i < 21*128; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);

  syr2k('l', 20, 129, -4.0, 3.5, A, B, C1);
  omp::syr2k('l', 20, 129, -4.0, 3.5, A, B, C2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(C1[i], C2[i]);
}


TEST_F(OMPBlasTest, Trmm) {

  Interval<double> A[10000], B2[10000], B1[10000];
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, B1);
  acopy(10000, ref->y, B2);


  trmm('l', 'u', 20, 129, 3.5, A, B1);
  omp::trmm('l', 'u', 20, 129, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);

  trmm('l', 'l', 20, 129, 3.5, A, B1);
  omp::trmm('l', 'l', 20, 129, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);

  trmm('R', 'u', 129, 20, 3.5, A, B1);
  omp::trmm('R', 'u', 129, 20, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);

  trmm('R', 'l', 129, 20, 3.5, A, B1);
  omp::trmm('R', 'l', 129, 20, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);
}


TEST_F(OMPBlasTest, Trsm) {

  Interval<double> A[10000], B2[10000], B1[10000];
  acopy(10000, ref->x, A);
  acopy(10000, ref->y, B1);
  acopy(10000, ref->y, B2);
  for(int i = 0; i < 20; ++i)
    A[20*i+i] = Interval<double>(i+3.0, i+30.0);


  trsm('l', 'u', 20, 23, 3.5, A, B1);
  omp::trsm('l', 'u', 20, 23, 3.5, A, B2);
  for(int i = 0; i < 20*23; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);

  trsm('l', 'l', 20, 129, 3.5, A, B1);
  omp::trsm('l', 'l', 20, 129, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);

  trsm('R', 'u', 129, 20, 3.5, A, B1);
  omp::trsm('R', 'u', 129, 20, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);

  trsm('R', 'l', 129, 20, 3.5, A, B1);
  omp::trsm('R', 'l', 129, 20, 3.5, A, B2);
  for(int i = 0; i < 20*129; ++i)
    EXPECT_FINTERVAL_EQ(B1[i], B2[i]);
}


#endif



