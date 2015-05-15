
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef SERIAL_BLAS2_TEST_CU
#define SERIAL_BLAS2_TEST_CU

#include <math.h>
#include "blas/serial_blas.h"
#include "aux/test_interval.h"
#include "aux/reference.h"
#include "gtest/gtest.h"

#include "aux/ulp.h"


using namespace intlag;


class SerialBlasTest : public ::testing::Test {
  protected:

    SerialBlasTest() {
      r = Reference::getInstance();
    }
    virtual ~SerialBlasTest(){}

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

    Reference* r;

    Interval<float> a[9], b[9], c[9];
    Interval<float> x[10000], y[10000], z[10000];
    Interval<double> da[9], db[9], dc[9], dx[10000], dy[10000], dz[10000];
};




/******************************** Tests ***************************************/

TEST_F(SerialBlasTest, Trans) {

  Interval<double> x[10000], xtrans[10000];
  acopy(10000, r->x, x);
  acopy(10000, r->x, xtrans);

  trans(34, 21, xtrans);
  for(int i = 0; i < 34; ++i)    
		for (int j = 0; j < 21; ++j)
      EXPECT_FINTERVAL_EQ(x[i*21+j], xtrans[j*34+i]);
}


TEST_F(SerialBlasTest, DiagonalUnit) {

  Interval<double> x[10000];
  acopy(10000, r->x, x);

  diagonal_unit(33, 34, x);
  for(int i = 0; i < 33; ++i)
    EXPECT_FINTERVAL_EQ(x[i*34+i], 1.0);
}


TEST_F(SerialBlasTest, Ger) {

  Interval<float> out[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  ger(2, 3, -2.2, a, b, out);
  EXPECT_FINTERVAL_EQ(out[0], Interval<float>(1.0));
  EXPECT_FINTERVAL_EQ(out[1], Interval<float>(2.0));
  EXPECT_FINTERVAL_EQ(out[2], Interval<float>(3.0));
  EXPECT_FINTERVAL_EQ(out[3], Interval<float>(4.0));
  EXPECT_FINTERVAL_EQ(out[4], Interval<float>(2.8, 7.2));
  EXPECT_FINTERVAL_EQ(out[5], Interval<float>(1.6, 10.4));
}


TEST_F(SerialBlasTest, Syr) {

  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, -400.0, 5.0, 6.0, -600.0, -300.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, -4.0, 27.0, 4.0, 5.0, -18.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, a, out_up);
  syr('U', 3, 2.2, a, out_sym_up);
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      EXPECT_FINTERVAL_EQ(out_up[i*3+j], out_sym_up[i*3+j]);

  ger(3, 3, 2.2, a, a, out_down);
  syr('l', 3, 2.2, a, out_sym_down);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j)
      EXPECT_FINTERVAL_EQ(out_down[i*3+j], out_sym_down[i*3+j]);
}


TEST_F(SerialBlasTest, Syr2) {

  Interval<float> zeroes[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> zeroes2[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, -400.0, 5.0, 6.0, -600.0, -300.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, -4.0, 27.0, 4.0, 5.0, -18.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, b, zeroes);
  ger(3, 3, 2.2, b, a, out_up);
  axpy(9, 1.0, zeroes, out_up);
  syr2('U', 3, 2.2, a, b, out_sym_up);
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      EXPECT_FINTERVAL_EQ(out_up[i*3+j], out_sym_up[i*3+j]);

  ger(3, 3, 2.2, a, b, zeroes2);
  ger(3, 3, 2.2, a, b, out_down);
  axpy(9, 1.0, zeroes2, out_down);
  syr2('l', 3, 2.2, a, b, out_sym_down);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j)
      EXPECT_FINTERVAL_EQ(out_down[i*3+j], out_sym_down[i*3+j]);
}


TEST_F(SerialBlasTest, Spr) {

  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, 5.0, 6.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, 4.0, 5.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, a, out_up);
  spr('u', 3, 2.2, a, out_sym_up);
  EXPECT_FINTERVAL_EQ(out_up[0], out_sym_up[0]);
  EXPECT_FINTERVAL_EQ(out_up[1], out_sym_up[1]);
  EXPECT_FINTERVAL_EQ(out_up[2], out_sym_up[2]);
  EXPECT_FINTERVAL_EQ(out_up[4], out_sym_up[3]);
  EXPECT_FINTERVAL_EQ(out_up[5], out_sym_up[4]);
  EXPECT_FINTERVAL_EQ(out_up[8], out_sym_up[5]);

  ger(3, 3, 2.2, a, a, out_down);
  spr('L', 3, 2.2, a, out_sym_down);
  EXPECT_FINTERVAL_EQ(out_down[0], out_sym_down[0]);
  EXPECT_FINTERVAL_EQ(out_down[3], out_sym_down[1]);
  EXPECT_FINTERVAL_EQ(out_down[4], out_sym_down[2]);
  EXPECT_FINTERVAL_EQ(out_down[6], out_sym_down[3]);
  EXPECT_FINTERVAL_EQ(out_down[7], out_sym_down[4]);
  EXPECT_FINTERVAL_EQ(out_down[8], out_sym_up[5]);
}


TEST_F(SerialBlasTest, Spr2) {

  Interval<float> zeroes[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> zeroes2[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Interval<float> out_up[] = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 6.0, 9.0};
  Interval<float> out_sym_up[] = {1.0, 2.0, 3.0, 5.0, 6.0, 9.0};
  Interval<float> out_down[] = {1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0};
  Interval<float> out_sym_down[] = {1.0, 4.0, 5.0, 7.0, 8.0, 9.0};

  ger(3, 3, 2.2, a, b, zeroes);
  ger(3, 3, 2.2, b, a, out_up);
  axpy(9, 1.0, zeroes, out_up);
  spr2('U', 3, 2.2, a, b, out_sym_up);
  EXPECT_FINTERVAL_EQ(out_up[0], out_sym_up[0]);
  EXPECT_FINTERVAL_EQ(out_up[1], out_sym_up[1]);
  EXPECT_FINTERVAL_EQ(out_up[2], out_sym_up[2]);
  EXPECT_FINTERVAL_EQ(out_up[4], out_sym_up[3]);
  EXPECT_FINTERVAL_EQ(out_up[5], out_sym_up[4]);
  EXPECT_FINTERVAL_EQ(out_up[8], out_sym_up[5]);

  ger(3, 3, 2.2, a, b, zeroes2);
  ger(3, 3, 2.2, a, b, out_down);
  axpy(9, 1.0, zeroes2, out_down);
  spr2('l', 3, 2.2, a, b, out_sym_down);
  EXPECT_FINTERVAL_EQ(out_down[0], out_sym_down[0]);
  EXPECT_FINTERVAL_EQ(out_down[3], out_sym_down[1]);
  EXPECT_FINTERVAL_EQ(out_down[4], out_sym_down[2]);
  EXPECT_FINTERVAL_EQ(out_down[6], out_sym_down[3]);
  EXPECT_FINTERVAL_EQ(out_down[7], out_sym_down[4]);
  EXPECT_FINTERVAL_EQ(out_down[8], out_sym_up[5]);

}


TEST_F(SerialBlasTest, Gemv) {

  Interval<float> out[3];

  acopy(3, b, out);
  gemv(1, 1, -2.5, 3.5, c, a, out);
  EXPECT_FINTERVAL_EQ(out[0], -2.5*c[0]*a[0]+3.5*b[0]);

  acopy(3, b, out);
  gemv(2, 2, Interval<float>(-2,2), 0.0, c, a, out);
  EXPECT_FINTERVAL_EQ(out[0], Interval<float>(-8.0,8.0));
  EXPECT_FINTERVAL_EQ(out[1], Interval<float>(-24.0,24.0));

  gemv(3, 3, 2.0, 3.0, c, a, b);
  EXPECT_FINTERVAL_EQ(b[0], Interval<float>(-20.0, 28.0));
  EXPECT_FINTERVAL_EQ(b[1], Interval<float>(-5.0, 163.0));
  EXPECT_FINTERVAL_EQ(b[2], Interval<float>(10.0, 406.0));
  
};


TEST_F(SerialBlasTest, Gbmv) {

  //Test 2 by 2
  Interval<float> out1[] = {1.1, 3.3};
  Interval<float> out11[] = {1.1, 3.3};
  Interval<float> matrix1[] = {1.0,3.0,0.0,2.0};
  Interval<float> banded1[] = {1.0,3.0,2.0,0.0};
  gemv(2, 2, -2.5, 3.5, matrix1, a, out1);
  gbmv(2, 2, 0, 1, -2.5, 3.5, banded1, a, out11);
  EXPECT_FINTERVAL_EQ(out1[0], out11[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out11[1]);

  //Test 3 by 3
  Interval<float> out2[3] = {1.1, 3.3, 1.3};
  Interval<float> out22[3] = {1.1, 3.3, 1.3};
  Interval<float> matrix2[] = {1.0,4.0,0.0,0.0,2.0,5.0,0.0,0.0,3.0};
  Interval<float> banded2[] = {1.0,4.0,2.0,5.0,3.0,-2222.0};
  gemv(3, 3, 22.5, 4.2, matrix2, b, out2);
  gbmv(3, 3, 0, 1, 22.5, 4.2, banded2, b, out22);
  EXPECT_FINTERVAL_EQ(out2[0], out22[0]);
  EXPECT_FINTERVAL_EQ(out2[1], out22[1]);
  EXPECT_FINTERVAL_EQ(out2[2], out22[2]);

  //Test 4 by 4
  Interval<double> out3[] = {-2.2, 5.5, 3.3, 20.3};
  Interval<double> out33[] = {-2.2, 5.5, 3.3, 20.3};
  Interval<double> matrix3[] = {1.0,4.0,0.0,0.0,5.0,2.0,5.0,0.0,2.0,7.0,3.0,-3.0,
0.0,3.0,5.0,-150.0};
  Interval<double> banded3[] = {0.0,0.0,1.0,4.0,0.0,5.0,2.0,5.0,2.0,7.0,3.0,-3.0,
3.0,5.0,-150.0,0.0};
  gemv(4, 4, -22.5, 4.2, matrix3, dc, out3);
  gbmv(4, 4, 2, 1, -22.5, 4.2, banded3, dc, out33);
  EXPECT_DINTERVAL_EQ(out3[0], out33[0]);
  EXPECT_DINTERVAL_EQ(out3[1], out33[1]);
  EXPECT_DINTERVAL_EQ(out3[2], out33[2]);
  EXPECT_DINTERVAL_EQ(out3[3], out33[3]);
}


TEST_F(SerialBlasTest, Symv) {

  //Test 2 by 2
  Interval<float> out1[] = {1.1, -3.3};
  Interval<float> out11[] = {1.1, -3.3};
  Interval<float> out111[] = {1.1, -3.3};
  Interval<float> matrix1[] = {1.0,3.0,3.0,3.3};
  gemv(2, 2, -2.5, 3.5, matrix1, a, out1);
  symv('U', 2, -2.5, 3.5, matrix1, a, out11);
  symv('L', 2, -2.5, 3.5, matrix1, a, out111);
  EXPECT_FINTERVAL_EQ(out1[0], out11[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out11[1]);
  EXPECT_FINTERVAL_EQ(out1[0], out111[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out111[1]);

  //Test 3 by 3
  Interval<float> out2[] = {35.35, 1.1, -3.3};
  Interval<float> out22[] = {35.35, 1.1, -3.3};
  Interval<float> out222[] = {35.35, 1.1, -3.3};
  Interval<float> matrix2[] = {1.0,-2.0,-5.0,-2.0,4.0,6.0,-5.0,6.0,-8.0};
  gemv(3, 3, -2.5, 3.5, matrix2, b, out2);
  symv('u', 3, -2.5, 3.5, matrix2, b, out22);
  symv('l', 3, -2.5, 3.5, matrix2, b, out222);
  EXPECT_FINTERVAL_EQ(out2[0], out22[0]);
  EXPECT_FINTERVAL_EQ(out2[1], out22[1]);
  EXPECT_FINTERVAL_EQ(out2[2], out22[2]);
  EXPECT_FINTERVAL_EQ(out2[0], out222[0]);
  EXPECT_FINTERVAL_EQ(out2[1], out222[1]);
  EXPECT_FINTERVAL_EQ(out2[2], out222[2]);
}


TEST_F(SerialBlasTest, Sbmv) {

  //Test 3 by 3
  Interval<float> out1[] = {35.35, 1.1, -3.3};
  Interval<float> out11[] = {35.35, 1.1, -3.3};
  Interval<float> out111[] = {35.35, 1.1, -3.3};
  Interval<float> matrix1[] = {1.0,4.0,0.0,4.0,2.0,5.0,0.0,5.0,3.0};
  Interval<float> banded1[] = {1.0,4.0,2.0,5.0,3.0,0.0};
  Interval<float> banded11[] = {0.0,1.0,4.0,2.0,5.0,3.0};
  gemv(3, 3, -2.5, 3.5, matrix1, b, out1);
  sbmv('u', 3, 1, -2.5, 3.5, banded1, b, out11);
  sbmv('l', 3, 1, -2.5, 3.5, banded11, b, out111);
  EXPECT_FINTERVAL_EQ(out1[0], out11[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out11[1]);
  EXPECT_FINTERVAL_EQ(out1[2], out11[2]);
  EXPECT_FINTERVAL_EQ(out1[0], out111[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out111[1]);
  EXPECT_FINTERVAL_EQ(out1[2], out111[2]);
}


TEST_F(SerialBlasTest, Spmv) {

  //Test 2 by 2
  Interval<float> out1[] = {35.35, 1.1};
  Interval<float> out11[] = {35.35, 1.1};
  Interval<float> out111[] = {35.35, 1.1};
  Interval<float> matrix1[] = {1.0,4.0,4.0,-2.0};
  Interval<float> compact1[] = {1.0,4.0,-2.0};
  gemv(2, 2, -2.5, 3.5, matrix1, c, out1);
  spmv('U', 2, -2.5, 3.5, compact1, c, out11);
  spmv('L', 2, -2.5, 3.5, compact1, c, out111);
  EXPECT_FINTERVAL_EQ(out1[0], out11[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out11[1]);
  EXPECT_FINTERVAL_EQ(out1[0], out111[0]);
  EXPECT_FINTERVAL_EQ(out1[1], out111[1]);

  //Test 4 by 4
  Interval<float> out2[] = {35.35, 1.1, -3.3, -10.0};
  Interval<float> out22[] = {35.35, 1.1, -3.3, -10.0};
  Interval<float> out222[] = {35.35, 1.1, -3.3, -10.0};
  Interval<float> matrix2[] = {1.0,5.0,6.0,7.0,5.0,2.0,8.0,9.0,6.0,8.0,3.0,10.0,
7.0,9.0,10.0,4.0};
  Interval<float> compact2[] = {1.0,5.0,6.0,7.0,2.0,8.0,9.0,3.0,10.0,4.0};
  Interval<float> compact22[] = {1.0,5.0,2.0,6.0,8.0,3.0,7.0,9.0,10.0,4.0};
  gemv(4, 4, -2.5, 3.5, matrix2, c, out2);
  spmv('U', 4, -2.5, 3.5, compact2, c, out22);
  spmv('L', 4, -2.5, 3.5, compact22, c, out222);
  EXPECT_FINTERVAL_EQ(out2[0], out22[0]);
  EXPECT_FINTERVAL_EQ(out2[1], out22[1]);
  EXPECT_FINTERVAL_EQ(out2[2], out22[2]);
  EXPECT_FINTERVAL_EQ(out2[3], out22[3]);
  EXPECT_FINTERVAL_EQ(out2[0], out222[0]);
  EXPECT_FINTERVAL_EQ(out2[1], out222[1]);
  EXPECT_FINTERVAL_EQ(out2[3], out222[3]);

}


TEST_F(SerialBlasTest, Trmv) {

  //Test 2 by 2
  Interval<float> out1[] = {35.35, 1.1};
  Interval<float> outup1[] = {35.35, 1.1};
  Interval<float> outdown1[] = {35.35, 1.1};
  Interval<float> trianup1[] = {1.0,2.0,0.0,4.0};
  Interval<float> triandown1[] = {1.0,0.0,-2.0,5.0};
  gemv(2, 2, 1.0, 0.0, trianup1, out1, c);
  trmv('u', 2, trianup1, outup1);
  EXPECT_FINTERVAL_EQ(c[0], outup1[0]);
  EXPECT_FINTERVAL_EQ(c[1], outup1[1]);

  gemv(2, 2, 1.0, 0.0, triandown1, out1, c);
  trmv('l', 2, triandown1, outdown1);
  EXPECT_FINTERVAL_EQ(c[0], outdown1[0]);
  EXPECT_FINTERVAL_EQ(c[1], outdown1[1]);

}


TEST_F(SerialBlasTest, Tbmv) {

  //Test 2 by 2
  Interval<float> out1[] = {35.35, 1.1};
  Interval<float> outup1[] = {35.35, 1.1};
  Interval<float> outdown1[] = {35.35, 1.1};
  Interval<float> trianup1[] = {1.0,2.0,0.0,4.0};
  Interval<float> bandedup1[] = {1.0,2.0,4.0,0.0};
  Interval<float> triandown1[] = {1.0,0.0,-2.0,5.0};
  Interval<float> bandeddown1[] = {0.0,1.0,-2.0,5.0};
  gemv(2, 2, 1.0, 0.0, trianup1, out1, c);
  tbmv('u', 2, 1, bandedup1, outup1);
  EXPECT_FINTERVAL_EQ(c[0], outup1[0]);
  EXPECT_FINTERVAL_EQ(c[1], outup1[1]);

  gemv(2, 2, 1.0, 0.0, triandown1, out1, c);
  tbmv('l', 2, 1, bandeddown1, outdown1);
  EXPECT_FINTERVAL_EQ(c[0], outdown1[0]);
  EXPECT_FINTERVAL_EQ(c[1], outdown1[1]);

}


TEST_F(SerialBlasTest, Tpmv) {

  //Test 2 by 2
  Interval<float> out1[] = {35.35, 1.1};
  Interval<float> outup1[] = {35.35, 1.1};
  Interval<float> outdown1[] = {35.35, 1.1};
  Interval<float> trianup1[] = {1.0,2.0,0.0,4.0};
  Interval<float> packedup1[] = {1.0,2.0,4.0};
  Interval<float> triandown1[] = {1.0,0.0,-2.0,5.0};
  Interval<float> packeddown1[] = {1.0,-2.0,5.0};
  gemv(2, 2, 1.0, 0.0, trianup1, out1, c);
  tpmv('U', 2, packedup1, outup1);
  EXPECT_FINTERVAL_EQ(c[0], outup1[0]);
  EXPECT_FINTERVAL_EQ(c[1], outup1[1]);

  gemv(2, 2, 1.0, 0.0, triandown1, out1, c);
  tpmv('l', 2, packeddown1, outdown1);
  EXPECT_FINTERVAL_EQ(c[0], outdown1[0]);
  EXPECT_FINTERVAL_EQ(c[1], outdown1[1]);

}


TEST_F(SerialBlasTest, Trsv) {

  //Test 2 by 2
  Interval<float> matrixup1[] = {33.0, 22.0, 0.0, -35.0};
  Interval<float> matrixdown1[] = {-35.0, 0.0, 33.0, 22.0};
  Interval<float> x1[] = {-35.35, -1.1};;
  Interval<float> x11[] = {-1.1, -35.35};
  trsv('U', 2, matrixup1, x11);
  EXPECT_FINTERVAL_EQ(x11[1], 35.35/35.0);
  EXPECT_FINTERVAL_EQ(x11[0], (-22.0*x11[1]-1.1)/33.0);
  trsv('L', 2, matrixdown1, x1);
  EXPECT_FINTERVAL_EQ(x1[0], -35.35/-35.0);
  EXPECT_FINTERVAL_EQ(x1[1], (-33.0*x1[0]-1.1)/22.0);

}


TEST_F(SerialBlasTest, Tbsv) {

  //Test 2 by 2
  Interval<float> matrixup1[] = {33.0, 22.0, 0.0, -35.0};
  Interval<float> bandedup1[] = {33.0, 22.0, -35.0, 0.0};
  Interval<float> matrixdown1[] = {-35.0, 0.0, 33.0, 22.0};
  Interval<float> bandeddown1[] = {0.0, -35.0, 33.0, 22.0};
  Interval<float> x1[] = {-35.35, -1.1};;
  Interval<float> x11[] = {-1.1, -35.35};
  Interval<float> b1[] = {-35.35, -1.1};;
  Interval<float> b11[] = {-1.1, -35.35};
  trsv('U', 2, matrixup1, x1);
  tbsv('U', 2, 1, bandedup1, b1);
  EXPECT_FINTERVAL_EQ(x1[0], b1[0]);
  EXPECT_FINTERVAL_EQ(x1[1], b1[1]);
  trsv('L', 2, matrixdown1, x11);
  tbsv('L', 2, 1, bandeddown1, b11);
  EXPECT_FINTERVAL_EQ(x11[0], b11[0]);
  EXPECT_FINTERVAL_EQ(x11[1], b11[1]);

}


TEST_F(SerialBlasTest, Tpsv) {


  //Test 2 by 2
  Interval<float> matrixup1[] = {33.0, 22.0, 0.0,-35.0};
  Interval<float> packedup1[] = {33.0, 22.0, -35.0};
  Interval<float> matrixdown1[] = {-35.0, 0.0, 33.0, 22.0};
  Interval<float> packeddown1[] = {-35.0, 33.0, 22.0};
  Interval<float> x1[] = {-35.35, -1.1};;
  Interval<float> x11[] = {-1.1, -35.35};
  Interval<float> b1[] = {-35.35, -1.1};;
  Interval<float> b11[] = {-1.1, -35.35};
  trsv('U', 2, matrixup1, x1);
  tpsv('U', 2, packedup1, b1);
  EXPECT_FINTERVAL_EQ(x1[0], b1[0]);
  EXPECT_FINTERVAL_EQ(x1[1], b1[1]);
  trsv('L', 2, matrixdown1, x11);
  tpsv('L', 2, packeddown1, b11);
  //EXPECT_FINTERVAL_EQ(x11[0], b11[0]);
  //EXPECT_FINTERVAL_EQ(x11[1], b11[1]);

}


#endif



