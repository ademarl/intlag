


#ifndef BOX_TEST
#define BOX_TEST

#include "stdio.h"
#include "box_blas.h"
#include "gtest/gtest.h"


class BoxTest : public ::testing::Test{

	protected:

	virtual void SetUp(){
		x = new interval_gpu<double>[2];
		x[0] = interval_gpu<double>(-11.5, 33.4);
		x[1] = interval_gpu<double>(22.2, 55.5);

		a = box_gpu<double>();
		b = box_gpu<double>(15);
		c = box_gpu<double>(x, 2);
		d = c;

	}

	virtual void TearDown(){
		delete[] x;
	}

	interval_gpu<double> *x;
	box_gpu<double> a, b, c, d;
};

TEST_F(BoxTest, BasicOps) {

	EXPECT_TRUE(a.empty());
	EXPECT_EQ(b.length(), 15);
	EXPECT_DOUBLE_EQ((c[0]).lower(), -11.5);
	EXPECT_DOUBLE_EQ((c.at(0)).upper(), 33.4);

	EXPECT_EQ(d.length(), 2);
	EXPECT_DOUBLE_EQ((d.at(0)).lower(), -11.5);
	EXPECT_DOUBLE_EQ((d[1]).upper(), 55.5);
};

TEST_F(BoxTest, Sum) {

	box_gpu<double> r = c + d;
	EXPECT_EQ(r.length(), 2);
	EXPECT_DOUBLE_EQ((r.at(0)).lower(), -23.0);
	EXPECT_DOUBLE_EQ((r[1]).upper(), 111.0);
}

TEST_F(BoxTest, LeftScalarMultiplication) {

	box_gpu<double> r = 3.7*c;
	EXPECT_EQ(r.length(), 2);
	EXPECT_DOUBLE_EQ((r[0]).lower(), -3.7*11.5);
	EXPECT_DOUBLE_EQ((r.at(1)).upper(), 3.7*55.5);
}

TEST_F(BoxTest, RightScalarMultiplication) {

	box_gpu<double> r = c*3.7;
	EXPECT_EQ(r.length(), 2);
	EXPECT_DOUBLE_EQ((r[0]).lower(), -3.7*11.5);
	EXPECT_DOUBLE_EQ((r.at(1)).upper(), 3.7*55.5);
}

TEST_F(BoxTest, Scal) {

	bool r = scal(300.7, c);
	EXPECT_EQ(c.length(), 2);
	EXPECT_TRUE(r);
	EXPECT_DOUBLE_EQ((c[0]).lower(), -300.7*11.5);
	EXPECT_DOUBLE_EQ((c.at(1)).upper(), 300.7*55.5);
}

TEST_F(BoxTest, ScalNegative) {

	bool r = scal(-5.5, c);
	EXPECT_EQ(c.length(), 2);
	EXPECT_TRUE(r);
	EXPECT_DOUBLE_EQ((c[0]).lower(), -5.5*33.4);
	EXPECT_DOUBLE_EQ((c.at(1)).upper(), -5.5*22.2);
}

TEST_F(BoxTest, AXPY) {

	bool r = axpy(3.7, c, c);
	EXPECT_TRUE(r);
	EXPECT_EQ(c.length(), 2);
	EXPECT_DOUBLE_EQ((c[0]).lower(), -4.7*11.5);
	EXPECT_DOUBLE_EQ((c.at(1)).upper(), 4.7*55.5);
}

TEST_F(BoxTest, ASumEven) {

	interval_gpu<double> s;
	bool r = asum(c, &s);
	EXPECT_TRUE(r);
	EXPECT_NEAR(s.lower(), -11.5 + 22.2, 0.1);
	EXPECT_NEAR(s.upper(), 33.4 + 55.5, 0.1);
}

TEST_F(BoxTest, ASumOdd) {
	interval_gpu<double> y[3];
	y[0] = interval_gpu<double>(-11.5, -1);
	y[1] = interval_gpu<double>(-1, 1);
	y[2] = interval_gpu<double>(33, 44);
	box_gpu<double> l(y, 3);

	interval_gpu<double> s;
	bool r = asum(l, &s);
	EXPECT_TRUE(r);
	EXPECT_NEAR(s.lower(), -11.5 - 1 + 33, 0.1);
	EXPECT_NEAR(s.upper(), -1 + 1 + 44, 0.1);
}

TEST_F(BoxTest, DotProduct) {

	interval_gpu<double> s;
	bool r = dot(c, ((double) 6)*c, &s);
	EXPECT_TRUE(r);
	EXPECT_NEAR(s.lower(), 6*(-11.5*33.4 + 22.2*22.2), 0.1);
	EXPECT_NEAR(s.upper(), 6*(33.4*33.4 + 55.5*55.5), 0.1);}

TEST_F(BoxTest, Norm2) {

	interval_gpu<double> s;
	bool r = norm2(c, &s);
	EXPECT_TRUE(r);
	EXPECT_NEAR(s.lower(), 10.427, 0.1);
	EXPECT_NEAR(s.upper(), 64.775, 0.1);
}

#endif
