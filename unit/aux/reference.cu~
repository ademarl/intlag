
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------



#include "reference.h"

// Defines a singleton class with random data inputs and reference results for all test operations


namespace intlag
{


Reference* reference = NULL;

void Reference::setValues(int argc, char** argv, int n){
  // Not thread safe
  if (reference == NULL) {
    if (n == 0 && argc == 1) reference = new Reference();
    else reference = new Reference(argc, argv, n);
  }
}


Reference* Reference::getInstance() {
  return reference;
}


//Generates general inputs
void Reference::general_init(){

      neg = Interval<double>(-3.1, -1.2);
      mid = Interval<double>(-5.1, 5.1);
      pos = Interval<double>(2.2, 4.4);

      zero = Interval<double>(0.0);
      point_pos = Interval<double>(2.0);
      point_neg = Interval<double>(-2.0);

      empty_i = Interval<double>();
      empty_down = Interval<double>(nan(""), 2048.8);
      empty_up = Interval<double>(-3000.8, nan(""));

      for(int i = 0; i < 9; ++i) {
          a[i] = Interval<double>(i);
          b[i] = Interval<double>(-i, i);
          c[i] = Interval<double>(i-5, i+i*i);
      }
}


// Generates a random number in [-MULTIPLIER, MULTIPLIER]
Interval<double> Reference::random_interval() {
  double x, y;
  x = (2*MULTIPLIER*rand())/RAND_MAX - MULTIPLIER;
  y = (2*MULTIPLIER*rand())/RAND_MAX - MULTIPLIER;
  if(x > y){
    double z = x;
    x = y;
    y = z;
  }
  return Interval<double>(x, y);
}


void Reference::random_init(int n){

  length = n;

  alpha = random_interval();
  beta = random_interval();

  x = (Interval<double>*) malloc(length*sizeof(Interval<double>));
  y = (Interval<double>*) malloc(length*sizeof(Interval<double>));
  if (x == NULL || y == NULL) {
    printf("Error, not enough memory for benchmark reference values!\n");
    exit(1);
  }

  for (int i = 0; i < length; ++i){
    x[i]  = random_interval();
    y[i]  = random_interval();
  }
}


void Reference::input_init(std::istream& is) {
  double aux, aux2;

  is >> length;

	is >> aux; is >> aux2;
	alpha = Interval<double>(aux, aux2);
	is >> aux; is >> aux2;
	beta = Interval<double>(aux, aux2);

  x = (Interval<double>*) malloc(length*sizeof(Interval<double>));
  y = (Interval<double>*) malloc(length*sizeof(Interval<double>));
  if (x == NULL || y == NULL) {
    printf("Error, not enough memory for benchmark reference values!\n");
    exit(1);
  }

  for (int i = 0; i < length; ++i){
    is >> aux; is >> aux2;
    x[i] = Interval<double>(aux, aux2);
	}
	for (int i = 0; i < length; ++i){
		is >> aux; is >> aux2;
    y[i] = Interval<double>(aux, aux2);
  }

}


// Initializes GPU by invoking CudaGeneral::scal
void Reference::benchCudaInit(Reference *r) {

  CudaInterval<double> *x = (CudaInterval<double>*) malloc((r->length)*sizeof(CudaInterval<double>));
  CudaInterval<double> alpha = r->alpha;
  acopy(r->length, r->x, x);
  DeviceData< CudaInterval<double> > dx(r->length, x);
  CudaGeneral::scal(r->length, alpha, dx.data());
  dx.toHost(x);
}


} // namespace intlag




