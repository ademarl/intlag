
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

// FIXME: Including this file twice?


#ifndef REFERENCE_H
#define REFERENCE_H

#include <iostream>
#include <fstream>
#include <cstdlib>

#include "../../include/interval/interval_lib.h"
#include "../../include/blas/cuda_blas.h"

#define MULTIPLIER 1.2

// Defines a singleton class with random data inputs and reference results for all test operations


namespace intlag
{


class Reference {

  Reference(){ length = 0; }

  Reference(int argc, char** argv, int n) {
    general_init();
    if (argc == 1) random_init(n);
    else {
      std::filebuf fb;
      if (fb.open (argv[1], std::ios::in)) {
        std::istream is(&fb);
        input_init(is);
        fb.close();
      }
    }
    benchCudaInit(this);
  }

  Interval<double> random_interval();
  void general_init();
  void random_init(int n);
  void input_init(std::istream& is);
  void benchCudaInit(Reference* r);


  public:

    ~Reference(){
      if (length == 0) return;

      free(x);
      free(y);
    }

    static void setValues(int argc, char** argv, int n);
    static Reference* getInstance();

    int length;
    Interval<double> alpha, beta;
    Interval<double> *x, *y;

    Interval<double> a[9], b[9], c[9];
    Interval<double> neg, mid, pos, zero, point_pos, point_neg, empty_i, empty_up, empty_down;
};


} // namespace intlag

#endif



