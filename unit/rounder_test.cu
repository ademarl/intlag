
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef ROUNDER_TEST_CU
#define ROUNDER_TEST_CU


#include <math.h>
#include "interval/rounder.h"
#include "gtest/gtest.h"

#include "aux/ulp.h"


using namespace intlag;

#define FLT_PRECISION 5e-6
#define DBL_PRECISION 5e-15


class RounderTest : public ::testing::Test {
  protected:

    RounderTest() {
      old_status = getRoundMode();
    }
    virtual ~RounderTest()
    {
    }

    virtual void SetUp()
    {
      x = 2.222222222;
      y = 3.333333333;
      a = 2.22222222222222222222;
      b = 3.33333333333333333333;

      fesetround(FE_DOWNWARD);
      sum = x+y;
      sub = x-y;
      mul = x*y;
      div = x/y;
      ma = fma(2.0f, x, y);
      sq   = sqrt(x);
      dsum = a+b;
      dsub = a-b;
      dmul = a*b;
      ddiv = a/b;
      dma = fma(2.0, a, b);
      dsq   = sqrt(a);
      setRoundMode(old_status);
    }
    virtual void TearDown()
    {
      setRoundMode(old_status);
    }


    short old_status;
    float x, y, sum, sub, mul, div, ma, sq;
    double a, b, dsum, dsub, dmul, ddiv, dma, dsq;
};




/********************************Tests ************/
TEST_F(RounderTest, Fenv) {

  setRoundMode(RoundStatus::up);
  ASSERT_NEAR(x+y, nextlarger(sum), FLT_PRECISION);
  ASSERT_NEAR(a+b, nextlarger(dsum), DBL_PRECISION);

  setRoundMode(RoundStatus::down);
  ASSERT_NEAR(x+y, sum, DBL_PRECISION);
  ASSERT_NEAR(a+b, dsum, DBL_PRECISION);
};

TEST_F(RounderTest, NaN) {

  ASSERT_TRUE(isnan(Rounder<float>::nan()));
  ASSERT_TRUE(isnan(Rounder<float>::nan()));
};

TEST_F(RounderTest, Sum) {

  ASSERT_NEAR(Rounder<float>::add_up(x, y), nextlarger(sum), FLT_PRECISION);
  ASSERT_NEAR(Rounder<float>::add_down(x, y), sum, FLT_PRECISION);
  ASSERT_NEAR(Rounder<double>::add_up(a,b), nextlarger(dsum), DBL_PRECISION);
  ASSERT_NEAR(Rounder<double>::add_down(a,b), dsum, DBL_PRECISION);
};

TEST_F(RounderTest, Sub) {

  ASSERT_NEAR(Rounder<float>::sub_up(x, y), nextlarger(sub), FLT_PRECISION);
  ASSERT_NEAR(Rounder<float>::sub_down(x, y), sub, FLT_PRECISION);
  ASSERT_NEAR(Rounder<double>::sub_up(a,b), nextlarger(dsub), DBL_PRECISION);
  ASSERT_NEAR(Rounder<double>::sub_down(a,b), dsub, DBL_PRECISION);
};

TEST_F(RounderTest, Mul) {

  ASSERT_NEAR(Rounder<float>::mul_up(x, y), nextlarger(mul), FLT_PRECISION);
  ASSERT_NEAR(Rounder<float>::mul_down(x, y), mul, FLT_PRECISION);
  ASSERT_NEAR(Rounder<double>::mul_up(a,b), nextlarger(dmul), DBL_PRECISION);
  ASSERT_NEAR(Rounder<double>::mul_down(a,b), dmul, DBL_PRECISION);
};

TEST_F(RounderTest, Div) {

  ASSERT_NEAR(Rounder<float>::div_up(x, y), nextlarger(div), FLT_PRECISION);
  ASSERT_NEAR(Rounder<float>::div_down(x, y), div, FLT_PRECISION);
  ASSERT_NEAR(Rounder<double>::div_up(a,b), nextlarger(ddiv), DBL_PRECISION);
  ASSERT_NEAR(Rounder<double>::div_down(a,b), ddiv, DBL_PRECISION);
};


TEST_F(RounderTest, Fma) {

  ASSERT_NEAR(Rounder<float>::fma_up(2.0, x, y), nextlarger(ma), FLT_PRECISION);
  ASSERT_NEAR(Rounder<float>::fma_down(2.0, x, y), ma, FLT_PRECISION);
  ASSERT_NEAR(Rounder<double>::fma_up(2.0, a,b), nextlarger(dma), DBL_PRECISION);
  ASSERT_NEAR(Rounder<double>::fma_down(2.0, a,b), dma, DBL_PRECISION);
};

TEST_F(RounderTest, Sqrt) {

  ASSERT_NEAR(Rounder<float>::sqrt_up(x), nextlarger(sq), FLT_PRECISION);
  ASSERT_NEAR(Rounder<float>::sqrt_down(x), sq, FLT_PRECISION);
  ASSERT_NEAR(Rounder<double>::sqrt_up(a), nextlarger(dsq), DBL_PRECISION);
  ASSERT_NEAR(Rounder<double>::sqrt_down(a), dsq, DBL_PRECISION);
};


#endif
