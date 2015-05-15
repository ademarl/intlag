
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef AUX_ULP_H
#define AUX_ULP_H


#include <math.h>

#define BIG_ENOUGH_NUMBER 9999999

template <class T>
T nextlarger(T x) {
  return nextafter(x, BIG_ENOUGH_NUMBER);
}

template <class T>
T nextsmaller(T x) {
  return nextafter(x, -BIG_ENOUGH_NUMBER);
}


#endif

