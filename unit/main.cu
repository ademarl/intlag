
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#include <stdio.h>
#include <ctime>

#include "gtest/gtest.h"
#include "aux/reference.h"

GTEST_API_ int main(int argc, char **argv) {

  int t = time(NULL);
  srand(t);
  intlag::Reference::setValues(argc, argv, 1000000);

  printf("Running main() from gtest_main.cc\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



