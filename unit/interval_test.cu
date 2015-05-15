
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef INTERVAL_TEST_CU
#define INTERVAL_TEST_CU

#include <math.h>
#include "interval/interval_lib.h"
#include "gtest/gtest.h"

#include "aux/reference.h"
#include "aux/test_interval.h"
#include "aux/ulp.h"

using namespace intlag;


class IntervalTest : public ::testing::Test {
  protected:

    IntervalTest() {
      r = Reference::getInstance();
    }
    virtual ~IntervalTest(){}

    virtual void SetUp()
    {
      neg = Interval<float>(-3.1, -1.2);
      mid = Interval<float>(-5.1, 5.1);
      pos = Interval<float>(2.2, 4.4);

      zero = Interval<float>(0.0);
      point_pos = Interval<float>(2.0);
      point_neg = Interval<float>(-2.0);

      empty_i = Interval<float>();
      empty_up = Interval<float>(nanf(""), 2048.8);
      empty_down = Interval<float>(-3000.8, nanf(""));


      dneg = Interval<double>(-3.1, -1.2);
      dmid = Interval<double>(-5.1, 5.1);
      dpos = Interval<double>(2.2, 4.4);

      dzero = Interval<double>(0.0);
      dpoint_pos = Interval<double>(2.0);
      dpoint_neg = Interval<double>(-2.0);

      dempty_i = Interval<double>();
      dempty_up = Interval<double>(nan(""), 2048.8);
      dempty_down = Interval<double>(-3000.8, nan(""));

      a = b = c = Interval<float>::empty();
      da = db = dc = Interval<double>::empty();
    }
    virtual void TearDown()
    {
    }

    Reference *r;

    
    Interval<float> z;
    Interval<float> a, b, c;
    Interval<double> da, db, dc;
    

    Interval<float> neg, mid, pos, zero, point_pos, point_neg, empty_i, empty_up, empty_down;
    Interval<double> dneg, dmid, dpos, dzero, dpoint_pos, dpoint_neg, dempty_i, dempty_up, dempty_down;
};




/******************************** Tests ***************************************/


TEST_F(IntervalTest, Constructors) {

  EXPECT_TRUE(isnan( r->empty_i.inf() ));
  EXPECT_TRUE(isnan( r->empty_i.sup() ));
  EXPECT_TRUE(isnan( (new Interval<float>)->inf() ));
  EXPECT_TRUE(isnan( (new Interval<float>)->sup() ));
  EXPECT_TRUE(isnan( (new Interval<double>)->inf() ));
  EXPECT_TRUE(isnan( (new Interval<double>)->sup() ));

  EXPECT_TRUE(isnan( r->empty_up.sup() ));
  EXPECT_TRUE(isnan( (z = r->empty_down).inf() ));

  EXPECT_DEATH(Interval<float>(150.1, 130.3), "");
  EXPECT_DEATH(Interval<float>(2.0, -3.0), "");
  EXPECT_DEATH(Interval<double>(-102.0, -300.5), "");

  EXPECT_FLOAT_EQ( (z = r->point_neg).inf(), -2.0);
  EXPECT_FLOAT_EQ( (z = r->point_pos).sup(), 2.0);
  EXPECT_FLOAT_EQ( (z = r->neg).inf(), -3.1);
  EXPECT_FLOAT_EQ( (z = r->pos).sup(), 4.4);

  EXPECT_DOUBLE_EQ(r->point_pos.inf(), 2.0);
  EXPECT_DOUBLE_EQ(r->point_neg.sup(), -2.0);
  EXPECT_DOUBLE_EQ(r->pos.inf(), 2.2);
  EXPECT_DOUBLE_EQ(r->neg.sup(), -1.2);
};


TEST_F(IntervalTest, ContainZero) {

EXPECT_TRUE(contain_zero(r->zero));
EXPECT_TRUE(contain_zero(z = r->mid));
EXPECT_FALSE(contain_zero(r->neg));
EXPECT_FALSE(contain_zero(z = r->pos));
}


TEST_F(IntervalTest, Empty) {

  // Method empty creates an empty interval
  EXPECT_TRUE(isnan( Interval<float>::empty().inf() )); // static call
  EXPECT_TRUE(isnan( Interval<double>::empty().sup() ));
  
  // Empty function, checks if an interval is empty
  EXPECT_TRUE(empty(r->mid.empty()));
  EXPECT_TRUE(empty(r->empty_i));
  EXPECT_TRUE(empty(z = r->empty_up));
  EXPECT_TRUE(empty(r->empty_down));
  EXPECT_FALSE(empty(z = r->neg));
  EXPECT_FALSE(empty(r->mid));
  EXPECT_FALSE(empty(r->pos));
};


TEST_F(IntervalTest, Equality) {

  EXPECT_TRUE(pos != mid);
  EXPECT_TRUE(pos != mid);
  EXPECT_FALSE(dpoint_pos == dpoint_neg);
  EXPECT_EQ(mid, Interval<float>(-5.1, 5.1));
  EXPECT_EQ(dmid, dmid);
  EXPECT_EQ(point_pos, 2.0);
  EXPECT_EQ(-2.0, point_neg);
}


TEST_F(IntervalTest, Attribution) {

  z = 2.0;
  EXPECT_EQ(z, Interval<float>(2.0));
  z = Interval<double>(3.5, 4.2);
  EXPECT_EQ(z, Interval<float>(3.5, 4.2));

}

TEST_F(IntervalTest, Width) {

  EXPECT_EQ( width(Interval<double>()), 0);
  EXPECT_FLOAT_EQ( width(Interval<float>(-3.1, -1.1)), 2.0);
  EXPECT_FLOAT_EQ( width(Interval<float>(-3.0, 11.0)), 14.0);
  EXPECT_DOUBLE_EQ( width(Interval<double>(9.999, 29.999)), 20.0);
};


TEST_F(IntervalTest, Median) {

  EXPECT_TRUE( isnan(median(Interval<double>())) );
  EXPECT_FLOAT_EQ( median(Interval<float>(-3, -1)), -2);
  EXPECT_FLOAT_EQ( median(Interval<float>(-3.0, 11.0)), 4.0);
  EXPECT_DOUBLE_EQ( median(Interval<double>(9.999, 29.999)), 19.999);

  EXPECT_TRUE( isnan(median_up(Interval<double>())) );
  EXPECT_FLOAT_EQ( median_up(Interval<float>(-3, -1)), -2);
  EXPECT_FLOAT_EQ( median_up(Interval<float>(-3.0, 11.0)), 4.0);
  EXPECT_DOUBLE_EQ( median_up(Interval<double>(9.999, 29.999)), 19.999);

  EXPECT_TRUE( isnan(median_down(Interval<double>())) );
  EXPECT_FLOAT_EQ( median_down(Interval<float>(-3, -1)), -2);
  EXPECT_FLOAT_EQ( median_down(Interval<float>(-3.0, 11.0)), 4.0);
  EXPECT_DOUBLE_EQ( median_down(Interval<double>(9.999, 29.999)), 19.999);
};


TEST_F(IntervalTest, Overlap) {

  EXPECT_EQ( overlap(neg, mid), neg);
  EXPECT_EQ( overlap(mid, neg), neg);
  EXPECT_EQ( overlap(dpos, dmid), dpos);
  
  EXPECT_EQ( overlap(dzero, dmid), dzero);
  EXPECT_TRUE( empty(overlap(dempty_up, dmid)));
  EXPECT_TRUE( empty(overlap(Interval<float>(5.2, 600.2), mid)));
};

TEST_F(IntervalTest, Hull) {

  EXPECT_EQ( hull(neg, mid), mid);
  EXPECT_EQ( hull(mid, neg), mid);
  EXPECT_EQ( hull(neg, pos), Interval<float>(-3.1, 4.4));
};


TEST_F(IntervalTest, Unary) {

  Interval<float> a(2.5, 2.7), b(-3.0, 4.2), c(-3.3, -3.2);

  EXPECT_EQ(a, +a);
  EXPECT_EQ(a, -(-a));
  EXPECT_EQ(b, +b);
  EXPECT_EQ(b, -(-b));
  EXPECT_EQ(c, +c);
  EXPECT_EQ(c, -(-c));
  EXPECT_EQ(-a, Interval<float>(-2.7, -2.5));
}


TEST_F(IntervalTest, Sum) {

  Interval<double> a(2.2), b(-3.0, 3.0), c;

  c = a+b;
  EXPECT_LE(c.inf(), c.sup());

  EXPECT_DOUBLE_EQ(c.inf(), -0.8);
  EXPECT_DOUBLE_EQ(c.sup(), 5.2);

  c = b+b;
  EXPECT_DOUBLE_EQ(c.inf(), -6.0);
  EXPECT_DOUBLE_EQ(c.sup(), 6.0);
}


TEST_F(IntervalTest, Sub) {

  Interval<double> a(-1.1, 2.2), b(-3.0, 5.0), c;

  c = a-b;
  EXPECT_LE(c.inf(), c.sup());

  EXPECT_DOUBLE_EQ(c.inf(), -6.1);
  EXPECT_DOUBLE_EQ(c.sup(), 5.2);

  c = b-b;
  EXPECT_DOUBLE_EQ(c.inf(), -8.0);
  EXPECT_DOUBLE_EQ(c.sup(), 8.0);
}


TEST_F(IntervalTest, MulScalLeft) {

  da = 2.0*dpos;
  EXPECT_LE(da.inf(), da.sup());
  EXPECT_DOUBLE_EQ(da.inf(), 2.0*2.2);
  EXPECT_DOUBLE_EQ(da.sup(), 2.0*4.4);

  db = -2.0*dmid;
  EXPECT_LE(db.inf(), db.sup());
  EXPECT_DOUBLE_EQ(db.inf(), -2.0*5.1);
  EXPECT_DOUBLE_EQ(db.sup(), 2.0*5.1);

  a = -2.0*neg;
  EXPECT_LE(a.inf(), a.sup());
  EXPECT_FLOAT_EQ(a.inf(), 2*1.2);
  EXPECT_FLOAT_EQ(a.sup(), 2*3.1);
}

TEST_F(IntervalTest, MulScalRight) {


  da = dpos*2.0;
  EXPECT_LE(da.inf(), da.sup());
  EXPECT_DOUBLE_EQ(da.inf(), 2.0*2.2);
  EXPECT_DOUBLE_EQ(da.sup(), 2.0*4.4);

  db = dmid*(-2.1);
  EXPECT_LE(db.inf(), db.sup());
  EXPECT_DOUBLE_EQ(db.inf(), -2.1*5.1);
  EXPECT_DOUBLE_EQ(db.sup(), 2.1*5.1);

  a = neg*(-2.2);
  EXPECT_LE(a.inf(), a.sup());
  EXPECT_FLOAT_EQ(a.inf(), 2.2*1.2);
  EXPECT_FLOAT_EQ(a.sup(), 2.2*3.1);
}


TEST_F(IntervalTest, Mul) {

  da = dpos*dpos;
  EXPECT_LE(da.inf(), da.sup());
  EXPECT_DOUBLE_EQ(da.inf(), 2.2*2.2);
  EXPECT_DOUBLE_EQ(da.sup(), 4.4*4.4);

  a = mid*mid;
  EXPECT_LE(a.inf(), a.sup());
  EXPECT_FLOAT_EQ(a.inf(), -5.1*5.1);
  EXPECT_FLOAT_EQ(a.sup(), 5.1*5.1);


  a = neg*pos;
  EXPECT_LE(a.inf(), a.sup());
  EXPECT_FLOAT_EQ(a.inf(), -3.1*4.4);
  EXPECT_FLOAT_EQ(a.sup(), -1.2*2.2);
}


TEST_F(IntervalTest, Fma) {

  da = fma(2.0, dpos, dpos);
  EXPECT_LE(da.inf(), da.sup());
  EXPECT_DOUBLE_EQ(da.inf(), 2*2.2+2.2);
  EXPECT_DOUBLE_EQ(da.sup(), 2*4.4+4.4);

  a = fma(2.0, mid, mid);
  EXPECT_LE(a.inf(), a.sup());
  EXPECT_FLOAT_EQ(a.inf(), -3*5.1);
  EXPECT_FLOAT_EQ(a.sup(), 3*5.1);


  a = fma(2.0, neg, pos);
  EXPECT_LE(a.inf(), a.sup());
  EXPECT_FLOAT_EQ(a.inf(), -2*3.1+2.2);
  EXPECT_FLOAT_EQ(a.sup(), -2*1.2+4.4);
}



TEST_F(IntervalTest, Div) {

  EXPECT_TRUE( isnan((neg/mid).inf()) );
  EXPECT_TRUE( isnan((pos/mid).inf()) );
  EXPECT_TRUE( isnan((dpos/dzero).inf()) );

  EXPECT_FLOAT_EQ( (neg/neg).inf(), 1.2/3.1 );
  EXPECT_FLOAT_EQ( (neg/neg).sup(), 3.1/1.2 );

  EXPECT_DOUBLE_EQ( (dpos/dpos).inf(), 2.2/4.4 );
  EXPECT_DOUBLE_EQ( (dpos/dpos).sup(), 4.4/2.2 );

  EXPECT_DOUBLE_EQ( (dneg/dpos).inf(), -3.1/2.2 );
  EXPECT_DOUBLE_EQ( (dneg/dpos).sup(), -1.2/4.4 );
}


TEST_F(IntervalTest, Sqrt) {

  Interval<double> a(2.2222222222222222222), b = sqrt(a);

  EXPECT_LT(b.inf(), b.sup());
  EXPECT_EQ(b.inf(), nextsmaller(b.sup()) );

  EXPECT_NEAR(b.inf(), 1.490712, 1e-5);
  EXPECT_NEAR(b.sup(), 1.490712, 1e-5);
}


TEST_F(IntervalTest, Abs) {

  a = abs(zero);
  b = abs(neg);
  c = abs(pos);
  EXPECT_EQ(a, Interval<float>(0, 0));
  EXPECT_EQ(b, Interval<float>(1.2, 3.1));
  EXPECT_EQ(c, Interval<float>(2.2, 4.4));
}


TEST_F(IntervalTest, CastAssignment) {

  a = dneg;
  b = dmid;
  da = pos;
  db = point_pos;

  EXPECT_EQ(a, Interval<float>(-3.1, -1.2));
  EXPECT_EQ(b, Interval<float>(-5.1, 5.1));
  EXPECT_EQ(da, Interval<double>(2.2f, 4.4f));
  EXPECT_EQ(db, Interval<double>(2.0f, 2.0f));
}



#endif



