
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef BOX_H
#define BOX_H

#include <assert.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "interval_lib.h"

using namespace std;

namespace intlag {


// FIXME: public members because attribution operation Box<T>[i] = x doesnt work
template <class T>
class Box{

  public:

		Box() {
      nl = nc = 0;
    }

		~Box() {
      elements.clear();
    }

    Box(unsigned int l) {
      nl = l; nc = 1;
      typename vector< Interval<T> >::iterator it = elements.begin();
      elements.insert(it, l, Interval<T>());
    }

    Box(unsigned int l, unsigned int c) {
      nl = l; nc = c;
      typename vector< Interval<T> >::iterator it = elements.begin();
      elements.insert(it, l*c, Interval<T>());
    }

    Box(unsigned int l, Interval<T> a) {
      nl = l; nc = 1;
      typename vector< Interval<T> >::iterator it = elements.begin();
      elements.insert(it, l, a);
    }

    Box(unsigned int l, unsigned int c, Interval<T> a) {
      nl = l; nc = c;
      typename vector< Interval<T> >::iterator it = elements.begin();
      elements.insert(it, l*c, a);
    }

    Box(unsigned int l, Interval<T> const *a) {
      nl = l; nc = 1;
      typename vector< Interval<T> >::iterator it = elements.begin();
      elements.insert(it, a, a+l);
    }

    Box(unsigned int l, unsigned int c, Interval<T> const *a) {
      nl = l; nc = c;
      typename vector< Interval<T> >::iterator it = elements.begin();
      elements.insert(it, a, a+l*c);
    }

    Box(const Box<T> &x) {
      nl = x.nl; nc = x.nc;
      elements = x.elements;
    }

    //Box(const Interval<T>* a, int n) {
		//	elements.assign(a, a+n);
    //}

		Interval<T>const* data() const {
			return &elements[0];
		}

		Interval<T>* data() {
			return &elements[0];
		}

    Interval<T> at (int n) {
      if (n >= (int)elements.size() || n < 0)
        throw out_of_range("Box index out of range.");
      return elements.at(n);
    }

		Interval<T> operator[] (int n) {
      if (n >= (int)elements.size() || n < 0)
        throw out_of_range("Box index out of range.");
			return elements[n];
		}

    int lins() const {
      return nl;
    }

    int cols() const {
      return nc;
    }

    int length() const {
      return elements.size();
    }

		bool empty() {
			return elements.empty();
		}

    template <class F>
    Box& operator=(const Interval<F> *rhs);
    template <class F>
    Box& operator=(const Box<F> &rhs);

	//private:
    unsigned int nl, nc;
		vector< Interval<T> > elements;
};


// Logical comparison
template <class T, class F> inline
bool operator==(const Box<T>& lhs, const Interval<F>* rhs)
{
  if (std::equal(lhs.elements.begin(), lhs.elements.end(), rhs))
    return true;
  return false;
}


template <class T, class F> inline
bool operator==(const Interval<T>* lhs, const Box<F>& rhs)
{
  if (std::equal(rhs.elements.begin(), rhs.elements.end(), lhs))
    return true;
  return false;
}


template <class T, class F> inline
bool operator==(const Box<T>& lhs, const Box<F>& rhs)
{
  if(lhs.lins() == rhs.lins() && lhs.cols() == rhs.cols())
    if (std::equal( lhs.elements.begin(), lhs.elements.end(), rhs.data()  ))
      return true;
  return false;
}


template <class T, class F> inline
bool operator!=(const Box<T>& lhs, const F& rhs)
{
  return !(lhs == rhs);
}

template <class T, class F> inline
bool operator!=(const F& lhs, const Box<T>& rhs)
{
  return !(lhs == rhs);
}


// Assignments

// Inference on the size of the array by the length of the vector
template <class T>
template <class F> inline
Box<T>& Box<T>::operator=(const Interval<F> *rhs)
{
  try {
    elements.assign(rhs, rhs + this->length());
  }
  catch(const std::exception& e) {
    throw out_of_range("Array index out of range in Box assignment.");
  }
  return *this;
}


template <class T>
template <class F> inline
Box<T>& Box<T>::operator=(const Box<F> &rhs)
{
    nl = rhs.nl; nc = rhs.nc;
    elements = rhs.elements;
    return *this;
}


} // namespace intlag

#endif



