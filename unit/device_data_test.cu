
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef DEVICE_DATA_TEST_CU
#define DEVICE_DATA_TEST_CU


#include "aux/device_data.h"
#include "gtest/gtest.h"

using namespace intlag;

TEST(DeviceDataTest, ToHost) {

	int z[4];
	int y[4] = {2, 3, 4, 110};
	DeviceData<int> x(4, y);
	x.toHost(z);

	EXPECT_EQ(2, z[0]);
	EXPECT_EQ(3, z[1]);
	EXPECT_EQ(4, z[2]);
	EXPECT_EQ(110, z[3]);
};

TEST(DeviceDataTest, Data) {

	double a[4] = {5, 6, 7, 42.42};
	double y[4] = {2.0, 3.0, 4.5, 111.1};
	DeviceData<double> x(4, y);
	CHECKED_CALL( cudaMemcpy(a, x.data(), 4*sizeof(double), cudaMemcpyDeviceToHost) );

	EXPECT_DOUBLE_EQ(2.0, a[0]);
	EXPECT_DOUBLE_EQ(3.0, a[1]);
	EXPECT_DOUBLE_EQ(4.5, a[2]);
	EXPECT_DOUBLE_EQ(111.1, a[3]);
};

TEST(DeviceDataTest, Swap) {

	double z[20000];
	double a[20000];
  for (int i = 0; i < 20000; ++i)
    a[i] = 2.31*i;

	DeviceData<double> x(20000, a), y(20000);
	swapByReference(x, y);
	y.toHost(z);

	EXPECT_DOUBLE_EQ(0, z[0]);
	EXPECT_DOUBLE_EQ(2.31, z[1]);
	EXPECT_DOUBLE_EQ(4.62, z[2]);
	EXPECT_DOUBLE_EQ(23100, z[10000]);
	EXPECT_DOUBLE_EQ(23102.31, z[10001]);
	EXPECT_DOUBLE_EQ(2.31*19999, z[19999]);
};

#endif
