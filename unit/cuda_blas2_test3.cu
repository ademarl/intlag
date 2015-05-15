
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef Cuda_BLAS2_TEST_CU
#define Cuda_BLAS2_TEST_CU

#include <math.h>
#include "blas/serial_blas.h"
#include "blas/omp_blas.h"
#include "aux/test_interval.h"
#include "aux/reference.h"
#include "aux/test_interval.h"
#include "gtest/gtest.h"


#include "blas/cuda_blas.h"

#include "aux/ulp.h"


using namespace intlag;


class CudaBlasTest : public ::testing::Test {
  protected:

    CudaBlasTest() {
      ref = Reference::getInstance();
    }
    virtual ~CudaBlasTest(){}

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

TEST_F(CudaBlasTest, CUDA) {

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


/******************************** Tests ***************************************/


TEST_F(CudaBlasTest, Trans) {

  Interval<double> x[10000], xtrans[10000];
  omp::acopy(10000, ref->x, x);
  omp::acopy(10000, ref->x, xtrans);

  omp::trans(34, 21, xtrans);
  for(int i = 0; i < 34; ++i)    
		for (int j = 0; j < 21; ++j)
      EXPECT_FINTERVAL_EQ(x[i*21+j], xtrans[j*34+i]);
}


TEST_F(CudaBlasTest, DiagonalUnit) {

  Interval<double> x[10000];
  omp::acopy(10000, ref->x, x);

  omp::diagonal_unit(33, 34, x);
  for(int i = 0; i < 33; ++i)
    EXPECT_FINTERVAL_EQ(x[i*34+i], 1.0);
}


TEST_F(CudaBlasTest, Ger) {

  Interval<float> out[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  omp::ger(2, 3, -2.2, a, b, out);
  EXPECT_FINTERVAL_EQ(out[0], Interval<float>(1.0));
  EXPECT_FINTERVAL_EQ(out[1], Interval<float>(2.0));
  EXPECT_FINTERVAL_EQ(out[2], Interval<float>(3.0));
  EXPECT_FINTERVAL_EQ(out[3], Interval<float>(4.0));
  EXPECT_FINTERVAL_EQ(out[4], Interval<float>(2.8, 7.2));
  EXPECT_FINTERVAL_EQ(out[5], Interval<float>(1.6, 10.4));
}


TEST_F(CudaBlasTest, Syr) {

  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, -400.0, 5.0, 6.0, -600.0, -300.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, -4.0, 27.0, 4.0, 5.0, -18.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, a, out_up);
  omp::syr('U', 3, 2.2, a, out_sym_up);
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      EXPECT_FINTERVAL_EQ(out_up[i*3+j], out_sym_up[i*3+j]);

  ger(3, 3, 2.2, a, a, out_down);
  omp::syr('l', 3, 2.2, a, out_sym_down);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j)
      EXPECT_FINTERVAL_EQ(out_down[i*3+j], out_sym_down[i*3+j]);
}


TEST_F(CudaBlasTest, Syr2) {

  Interval<float> zeroes[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> zeroes2[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, -400.0, 5.0, 6.0, -600.0, -300.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, -4.0, 27.0, 4.0, 5.0, -18.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, b, zeroes);
  ger(3, 3, 2.2, b, a, out_up);
  axpy(9, 1.0, zeroes, out_up);
  omp::syr2('U', 3, 2.2, a, b, out_sym_up);
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      EXPECT_FINTERVAL_EQ(out_up[i*3+j], out_sym_up[i*3+j]);

  ger(3, 3, 2.2, a, b, zeroes2);
  ger(3, 3, 2.2, a, b, out_down);
  axpy(9, 1.0, zeroes2, out_down);
  omp::syr2('l', 3, 2.2, a, b, out_sym_down);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j)
      EXPECT_FINTERVAL_EQ(out_down[i*3+j], out_sym_down[i*3+j]);
}


TEST_F(CudaBlasTest, Spr) {

  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, 5.0, 6.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, 4.0, 5.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, a, out_up);
  omp::spr('u', 3, 2.2, a, out_sym_up);
  EXPECT_FINTERVAL_EQ(out_up[0], out_sym_up[0]);
  EXPECT_FINTERVAL_EQ(out_up[1], out_sym_up[1]);
  EXPECT_FINTERVAL_EQ(out_up[2], out_sym_up[2]);
  EXPECT_FINTERVAL_EQ(out_up[4], out_sym_up[3]);
  EXPECT_FINTERVAL_EQ(out_up[5], out_sym_up[4]);
  EXPECT_FINTERVAL_EQ(out_up[8], out_sym_up[5]);

  ger(3, 3, 2.2, a, a, out_down);
  omp::spr('L', 3, 2.2, a, out_sym_down);
  EXPECT_FINTERVAL_EQ(out_down[0], out_sym_down[0]);
  EXPECT_FINTERVAL_EQ(out_down[3], out_sym_down[1]);
  EXPECT_FINTERVAL_EQ(out_down[4], out_sym_down[2]);
  EXPECT_FINTERVAL_EQ(out_down[6], out_sym_down[3]);
  EXPECT_FINTERVAL_EQ(out_down[7], out_sym_down[4]);
  EXPECT_FINTERVAL_EQ(out_down[8], out_sym_up[5]);
}


TEST_F(CudaBlasTest, Spr2) {

  Interval<float> zeroes[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> zeroes2[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, 5.0, 6.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, 4.0, 5.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, b, zeroes);
  ger(3, 3, 2.2, b, a, out_up);
  axpy(9, 1.0, zeroes, out_up);
  omp::spr2('U', 3, 2.2, a, b, out_sym_up);
  EXPECT_FINTERVAL_EQ(out_up[0], out_sym_up[0]);
  EXPECT_FINTERVAL_EQ(out_up[1], out_sym_up[1]);
  EXPECT_FINTERVAL_EQ(out_up[2], out_sym_up[2]);
  EXPECT_FINTERVAL_EQ(out_up[4], out_sym_up[3]);
  EXPECT_FINTERVAL_EQ(out_up[5], out_sym_up[4]);
  EXPECT_FINTERVAL_EQ(out_up[8], out_sym_up[5]);

  ger(3, 3, 2.2, a, b, zeroes2);
  ger(3, 3, 2.2, a, b, out_down);
  axpy(9, 1.0, zeroes2, out_down);
  omp::spr2('l', 3, 2.2, a, b, out_sym_down);
  EXPECT_FINTERVAL_EQ(out_down[0], out_sym_down[0]);
  EXPECT_FINTERVAL_EQ(out_down[3], out_sym_down[1]);
  EXPECT_FINTERVAL_EQ(out_down[4], out_sym_down[2]);
  EXPECT_FINTERVAL_EQ(out_down[6], out_sym_down[3]);
  EXPECT_FINTERVAL_EQ(out_down[7], out_sym_down[4]);
  EXPECT_FINTERVAL_EQ(out_down[8], out_sym_up[5]);

}


TEST_F(CudaBlasTest, Gemv) {

  Interval<float> out[3];

  omp::acopy(3, b, out);
  omp::gemv(1, 1, -2.5, 3.5, c, a, out);
  EXPECT_FINTERVAL_EQ(out[0], -2.5*c[0]*a[0]+3.5*b[0]);

  omp::acopy(3, b, out);
  omp::gemv(2, 2, Interval<float>(-2,2), 0.0, c, a, out);
  EXPECT_FINTERVAL_EQ(out[0], Interval<float>(-8.0,8.0));
  EXPECT_FINTERVAL_EQ(out[1], Interval<float>(-24.0,24.0));

  omp::gemv(3, 3, 2.0, 3.0, c, a, b);
  EXPECT_FINTERVAL_EQ(b[0], Interval<float>(-20.0, 28.0));
  EXPECT_FINTERVAL_EQ(b[1], Interval<float>(-5.0, 163.0));
  EXPECT_FINTERVAL_EQ(b[2], Interval<float>(10.0, 406.0));
  
  Interval<double> x[10000], y[10000], z[10000], w[10000];
  omp::acopy(10000, ref->x, x);
  omp::acopy(10000, ref->y, y);
  omp::acopy(10000, ref->x, z);
  omp::acopy(10000, ref->x, w);

  gemv(340, 27, 4.444, -3.5, x, y, w);
  omp::gemv(340, 27, 4.444, -3.5, x, y, z);

  for(int i = 0; i < 340; ++i)
    EXPECT_FINTERVAL_EQ(z[i], w[i]);
  for(int i = 340; i < 10000; ++i)
    EXPECT_FINTERVAL_EQ(z[i], x[i]);
};


TEST_F(CudaBlasTest, Gbmv) {

  Interval<double> x[10000], A[10000], y1[10000], y2[10000];
  omp::acopy(10000, ref->x, x);
  omp::acopy(10000, ref->y, y1);
  omp::acopy(10000, ref->y, y2);
  omp::acopy(10000, ref->x, A);

  gbmv(33, 22, 5, 8, 22.5, 4.2, A, x, y1);
  omp::gbmv(33, 22, 5, 8, 22.5, 4.2, A, x, y2);

  for(int i = 0; i < 33; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);
}


TEST_F(CudaBlasTest, Symv) {

  Interval<double> x[10000], A[10000], y1[10000], y2[10000];
  omp::acopy(10000, ref->x, x);
  omp::acopy(10000, ref->y, y1);
  omp::acopy(10000, ref->y, y2);
  omp::acopy(10000, ref->x, A);

  symv('U', 23, 2.5, 33.54, A, x, y1);
  omp::symv('u', 23, 2.5, 33.54, A, x, y2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);

  symv('l', 23, 2.5, 33.54, A, x, y1);
  omp::symv('L', 23, 2.5, 33.54, A, x, y2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);

}


TEST_F(CudaBlasTest, Sbmv) {

  Interval<double> x[10000], A[10000], y1[10000], y2[10000];
  omp::acopy(10000, ref->x, x);
  omp::acopy(10000, ref->y, y1);
  omp::acopy(10000, ref->y, y2);
  omp::acopy(10000, ref->x, A);

  sbmv('U', 22, 5, 2.5, 33.54, A, x, y1);
  omp::sbmv('u', 22, 5, 2.5, 33.54, A, x, y2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);

  sbmv('L', 22, 5, 2.5, 33.54, A, x, y1);
  omp::sbmv('l', 22, 5, 2.5, 33.54, A, x, y2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);
}


TEST_F(CudaBlasTest, Spmv) {
  Interval<double> x[10000], A[10000], y1[10000], y2[10000];
  omp::acopy(10000, ref->x, x);
  omp::acopy(10000, ref->y, y1);
  omp::acopy(10000, ref->y, y2);
  omp::acopy(10000, ref->x, A);

  spmv('U', 22, 2.5, 33.54, A, x, y1);
  omp::spmv('u', 22, 2.5, 33.54, A, x, y2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);

  spmv('L', 22, 2.5, 33.54, A, x, y1);
  omp::spmv('l', 22, 2.5, 33.54, A, x, y2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(y1[i], y2[i]);
}


TEST_F(CudaBlasTest, Trmv) {
  Interval<double> A[10000], x1[10000], x2[10000];
  omp::acopy(10000, ref->x, x1);
  omp::acopy(10000, ref->x, x2);
  omp::acopy(10000, ref->y, A);

  trmv('U', 22, A, x1);
  omp::trmv('u', 22, A, x2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

  trmv('L', 23, A, x1);
  omp::trmv('l', 23, A, x2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


TEST_F(CudaBlasTest, Tbmv) {
  Interval<double> A[10000], x1[10000], x2[10000];
  omp::acopy(10000, ref->x, x1);
  omp::acopy(10000, ref->x, x2);
  omp::acopy(10000, ref->y, A);

  tbmv('U', 22, 3, A, x1);
  omp::tbmv('u', 22, 3, A, x2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

  tbmv('L', 23, 5, A, x1);
  omp::tbmv('l', 23, 5, A, x2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


TEST_F(CudaBlasTest, Tpmv) {
  Interval<double> A[10000], x1[10000], x2[10000];
  omp::acopy(10000, ref->x, x1);
  omp::acopy(10000, ref->x, x2);
  omp::acopy(10000, ref->y, A);

  tpmv('U', 22, A, x1);
  omp::tpmv('u', 22, A, x2);
  for(int i = 0; i < 22; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

  tpmv('L', 23, A, x1);
  omp::tpmv('l', 23, A, x2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


TEST_F(CudaBlasTest, Trsv) {
  Interval<double> A[10000], x1[10000], x2[10000];
  omp::acopy(10000, ref->x, x1);
  omp::acopy(10000, ref->x, x2);
  omp::acopy(10000, ref->y, A);

  // A diagonal must not contain 0
  for(int i = 0; i < 23; ++i)
    A[23*i+i] = Interval<double>(i+3.0, i+30.0);

  trsv('U', 23, A, x1);
  omp::trsv('u', 23, A, x2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);


  for(int i = 0; i < 24; ++i)
    A[24*i+i] = Interval<double>(-(i+34.0), -(i+4.0));
  trsv('L', 24, A, x1);
  omp::trsv('l', 24, A, x2);
  for(int i = 0; i < 24; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


TEST_F(CudaBlasTest, Tbsv) {
  Interval<double> A[10000], x1[10000], x2[10000];
  omp::acopy(10000, ref->x, x1);
  omp::acopy(10000, ref->x, x2);
  omp::acopy(10000, ref->y, A);

  // A diagonal must not contain 0
  for(int i = 0; i < 23; ++i)
    A[5*i+0] = Interval<double>(i+3.0, i+30.0);

  tbsv('U', 23, 4, A, x1);
  omp::tbsv('u', 23, 4, A, x2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);


  for(int i = 0; i < 24; ++i)
    A[5*i+4] = Interval<double>(-(i+34.0), -(i+4.0));
  tbsv('L', 24, 4, A, x1);
  omp::tbsv('l', 24, 4, A, x2);
  for(int i = 0; i < 24; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);
}


TEST_F(CudaBlasTest, Tpsv) {

  Interval<double> A[10000], x1[10000], x2[10000];
  omp::acopy(10000, ref->x, x1);
  omp::acopy(10000, ref->x, x2);
  omp::acopy(10000, ref->y, A);

  // A diagonal must not contain 0
  for(int i = 0; i < 23; ++i)
    A[INDEX_TRIAN_UP(23, i, i)] = Interval<double>(-(i+34.0), -(i+4.0));

  tpsv('U', 23, A, x1);
  omp::tpsv('u', 23, A, x2);
  for(int i = 0; i < 23; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);


  for(int i = 0; i < 24; ++i)
    A[INDEX_TRIAN_DOWN(24, i, i)] = Interval<double>(i+3.0, i+30.0);
  tpsv('L', 24, A, x1);
  omp::tpsv('l', 24, A, x2);
  for(int i = 0; i < 24; ++i)
    EXPECT_DINTERVAL_EQ(x1[i], x2[i]);

}


#endif



