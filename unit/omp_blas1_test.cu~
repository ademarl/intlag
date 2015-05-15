
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef OMP_BLAS1_TEST_CU
#define OMP_BLAS1_TEST_CU

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


TEST_F(OMPBlasTest, ACopy) {

	omp::acopy(3, x, z);
  omp::acopy(3, dx, dz);

  for(int i = 0; i < 3; ++i){
    EXPECT_FINTERVAL_EQ(z[i], x[i]);
    EXPECT_DINTERVAL_EQ(dz[i], dx[i]);
  }
}


TEST_F(OMPBlasTest, Swap) {

	omp::swap(3, x, z);
	omp::swap(3, dx, dy);

  for(int i = 0; i < 3; ++i){
    EXPECT_FINTERVAL_EQ(z[i], Interval<float>(i));
    EXPECT_FINTERVAL_EQ(x[i], Interval<float>(i-5000, i+i*i));
    EXPECT_DINTERVAL_EQ(dy[i], Interval<double>(i));
    EXPECT_DINTERVAL_EQ(dx[i], Interval<double>(-i, i));
  }
}

TEST_F(OMPBlasTest, Scal) {

  omp::scal(9, -2.0, a);
  omp::scal(9, -2.3, b);
  omp::scal(5005, 3.5, dz);
  for(int i = 0; i < 9; ++i) {
    EXPECT_FINTERVAL_EQ(a[i], Interval<float>(-2.0*i));
    EXPECT_FINTERVAL_EQ(b[i], Interval<float>(-2.3*i, 2.3*i));
  }

  for(int i = 0; i < 5000; ++i)
    EXPECT_DINTERVAL_EQ(dz[i], Interval<double>(3.5*(i-5000), 3.5*(i+i*i)));
  EXPECT_DINTERVAL_EQ(dz[5006], Interval<double>(6.0, (5006.0+5006.0*5006.0)));
};


TEST_F(OMPBlasTest, AXPY) {

  omp::axpy(9, 18.4, b, a);
  omp::axpy(9, -2.2, b, c);
  omp::axpy(10000, -3.3, dx, dy); 

  for(int i = 0; i < 9; ++i) {
    EXPECT_FINTERVAL_EQ(a[i], Interval<float>(-18.4*i+i, 18.4*i+i));
    EXPECT_FINTERVAL_EQ(c[i], Interval<float>(-2.2*i+i-5, 2.2*i+i+i*i));
  }

  for(int i = 0; i < 10000; ++i) {
    EXPECT_DINTERVAL_EQ(dy[i], Interval<double>(-3.3*i-i, -3.3*i+i));
  }

};


TEST_F(OMPBlasTest, ASum) {

  omp::asum(9, c, b);
  asum(9, a, b);
  omp::asum(10000, dz, dx);
  asum(10000, dy, dx);

  EXPECT_FINTERVAL_EQ(c[0], Interval<float>(0.0, 36.0));
  EXPECT_FINTERVAL_EQ(c[0], a[0]);
  EXPECT_DINTERVAL_EQ(dz[0], Interval<double>(10000*9999/2.0));
  EXPECT_DINTERVAL_EQ(dz[0], dy[0]);

};


TEST_F(OMPBlasTest, Dot) {

  omp::dot(9, c, a, b);
  dot(9, x, a, b);
  EXPECT_FINTERVAL_EQ(c[0], Interval<float>(0.0, 204.0));
  EXPECT_FINTERVAL_EQ(c[0], x[0]);
};


TEST_F(OMPBlasTest, Norm2) {

  Interval<float> g(-35.0, 9.0), h(-2.0, 3.0), f(1.2, 2.4);

  omp::norm2(1, c, &g);
  EXPECT_FINTERVAL_EQ(c[0], Interval<float>(0.0, 35.0));
  omp::norm2(1, c, &h);
  EXPECT_FINTERVAL_EQ(c[0], Interval<float>(0.0, 3.0));
  omp::norm2(1, c, &f);
  EXPECT_FINTERVAL_EQ(c[0], f);

  omp::norm2(9, c, a);
  EXPECT_FINTERVAL_EQ(c[0], 14.282857);

  norm2(29, dz, dx);
  omp::norm2(29, dy, dx);
  EXPECT_NEAR(dz[0].inf(), dy[0].inf(), 1e-6);
  EXPECT_NEAR(dz[0].sup(), dy[0].sup(), 1e-6);
};


TEST_F(OMPBlasTest, Rot) {

  omp::rot(3, a, b, 1.0, 0.0);
  EXPECT_FINTERVAL_EQ(a[0], 0.0);
  EXPECT_FINTERVAL_EQ(a[1], 1.0);
  EXPECT_FINTERVAL_EQ(a[2], 2.0);
  EXPECT_FINTERVAL_EQ(b[0], 0.0);
  EXPECT_FINTERVAL_EQ(b[1], Interval<float>(-1.0, 1.0));
  EXPECT_FINTERVAL_EQ(b[2], Interval<float>(-2.0, 2.0));

  omp::rot(3, da, dc, 0.0, -1.0);
  EXPECT_DINTERVAL_EQ(dc[0], 0.0);
  EXPECT_DINTERVAL_EQ(dc[1], 1.0);
  EXPECT_DINTERVAL_EQ(dc[2], 2.0);
  EXPECT_DINTERVAL_EQ(da[0], Interval<double>(0, 5.0));
  EXPECT_DINTERVAL_EQ(da[1], Interval<double>(-2.0, 4.0));
  EXPECT_DINTERVAL_EQ(da[2], Interval<double>(-6.0, 3.0));

  omp::rot(2, b, c, 2.0, 2.0);
  EXPECT_FINTERVAL_EQ(b[0], Interval<float>(-10.0, 0.0));
  EXPECT_FINTERVAL_EQ(b[1], Interval<float>(-10.0, 6.0));
  EXPECT_FINTERVAL_EQ(c[0], Interval<float>(-10.0, 0.0));
  EXPECT_FINTERVAL_EQ(c[1], Interval<float>(-10.0, 6.0));
};


TEST_F(OMPBlasTest, Rotm) {

  int H2[] = {-2,15,16,17,-18};
  omp::rotm(3, da, db, H2);
  EXPECT_DINTERVAL_EQ(da[0], 0.0);
  EXPECT_DINTERVAL_EQ(da[1], 1.0);
  EXPECT_DINTERVAL_EQ(da[2], 2.0);
  EXPECT_DINTERVAL_EQ(db[0], Interval<double>(-0, 0));
  EXPECT_DINTERVAL_EQ(db[1], Interval<double>(-1.0, 1.0));
  EXPECT_DINTERVAL_EQ(db[2], Interval<double>(-2.0, 2.0));

  int H1[] = {1,0,0,0,0};
  omp::rotm(3, x, y, H1);
  EXPECT_FINTERVAL_EQ(x[0], 0.0);
  EXPECT_FINTERVAL_EQ(x[1], Interval<float>(-1.0, 1.0));
  EXPECT_FINTERVAL_EQ(y[0], 0.0);
  EXPECT_FINTERVAL_EQ(y[1], -1.0);

  int H0[] = {0,0,0,0,0};
  omp::rotm(2, dx, dy, H0);
  EXPECT_DINTERVAL_EQ(dx[0], 0.0);
  EXPECT_DINTERVAL_EQ(dx[1], 1.0);
  EXPECT_DINTERVAL_EQ(dx[2], 2.0);
  EXPECT_DINTERVAL_EQ(dy[0], Interval<double>(-0, 0));
  EXPECT_DINTERVAL_EQ(dy[1], Interval<double>(-1.0, 1.0));
  EXPECT_DINTERVAL_EQ(dy[2], Interval<double>(-2.0, 2.0));

  int H[] = {-1,0,0,0,0};
  omp::rotm(3, a, b, H);
  EXPECT_FINTERVAL_EQ(a[0], 0.0);
  EXPECT_FINTERVAL_EQ(a[1], 0.0);
  EXPECT_FINTERVAL_EQ(a[2], 0.0);
  EXPECT_FINTERVAL_EQ(b[0], 0.0);
  EXPECT_FINTERVAL_EQ(b[1], 0.0);
  EXPECT_FINTERVAL_EQ(b[2], 0.0);
}


#endif



