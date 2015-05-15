
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#include <math.h>
#include "min_max.h"

namespace intlag {


float min(float a, float b)
{
  return fminf(a,b);
}

double min(double a, double b)
{
  return fmin(a,b);
}

float max(float a, float b)
{
  return fmaxf(a,b);
}

double max(double a, double b)
{
  return fmax(a,b);
}

float min(float a, float b, float c, float d)
{
  return fminf(fminf(a,b), fminf(c,d));
}

double min(double a, double b, double c, double d)
{
  return fmin(fmin(a,b), fmin(c,d));
}

float max(float a, float b, float c, float d)
{
  return fmaxf(fmaxf(a,b), fmaxf(c,d));
}

double max(double a, double b, double c, double d)
{
  return fmax(fmax(a,b), fmax(c,d));
}


} // namespace intlag



