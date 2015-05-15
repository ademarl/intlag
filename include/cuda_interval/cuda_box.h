
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BOX_H
#define CUDA_BOX_H

#include <assert.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "cuda_interval_lib.h"

using namespace std;

namespace intlag {

// FIXME: public members because attribution operation Box<T>[i] = x doesnt work
template <class T>
class CudaBox{

  public:

		__host__ CudaBox() {
      nl = nc = 0;
    }

		__host__ ~CudaBox() {
      elements.clear();
    }

    __host__ CudaBox(unsigned int l) {
      nl = l; nc = 1;
      typename vector< CudaInterval<T> >::iterator it = elements.begin();
      elements.insert(it, l, CudaInterval<T>());
    }

    __host__ CudaBox(unsigned int l, unsigned int c) {
      nl = l; nc = c;
      typename vector< CudaInterval<T> >::iterator it = elements.begin();
      elements.insert(it, l*c, CudaInterval<T>());
    }

    __host__ CudaBox(unsigned int l, CudaInterval<T> a) {
      nl = l; nc = 1;
      typename vector< CudaInterval<T> >::iterator it = elements.begin();
      elements.insert(it, l, a);
    }

    __host__ CudaBox(unsigned int l, unsigned int c, CudaInterval<T> a) {
      nl = l; nc = c;
      typename vector< CudaInterval<T> >::iterator it = elements.begin();
      elements.insert(it, l*c, a);
    }

    __host__ CudaBox(unsigned int l, CudaInterval<T> const *a) {
      nl = l; nc = 1;
      typename vector< CudaInterval<T> >::iterator it = elements.begin();
      elements.insert(it, a, a+l);
    }

    __host__ CudaBox(unsigned int l, unsigned int c, CudaInterval<T> const *a) {
      nl = l; nc = c;
      typename vector< CudaInterval<T> >::iterator it = elements.begin();
      elements.insert(it, a, a+l*c);
    }

    __host__ CudaBox(const CudaBox<T> &x) {
      nl = x.nl; nc = x.nc;
      elements = x.elements;
    }

    //CudaBox(const CudaInterval<T>* a, int n) {
		//	elements.assign(a, a+n);
    //}

		__host__ CudaInterval<T>const* data() const {
			return &elements[0];
		}

		__host__ CudaInterval<T>* data() {
			return &elements[0];
		}

    __host__ CudaInterval<T> at (int n) {
      if (n >= (int)elements.size() || n < 0)
        throw out_of_range("CudaBox index out of range.");
      return elements.at(n);
    }

		__host__ CudaInterval<T> operator[] (int n) {
      if (n >= (int)elements.size() || n < 0)
        throw out_of_range("CudaBox index out of range.");
			return elements[n];
		}

    __host__ int lins() const {
      return nl;
    }

    __host__ int cols() const {
      return nc;
    }

    __host__ int length() const {
      return elements.size();
    }

		__host__ bool empty() {
			return elements.empty();
		}

    template <class F> __host__ 
    CudaBox& operator=(const CudaInterval<F> *rhs);
    template <class F> __host__ 
    CudaBox& operator=(const CudaBox<F> &rhs);

	//private:
    unsigned int nl, nc;
		vector< CudaInterval<T> > elements;
};


// Logical comparison
template <class T, class F> inline __host__ 
bool operator==(const CudaBox<T>& lhs, const CudaInterval<F>* rhs)
{
  if (std::equal(lhs.elements.begin(), lhs.elements.end(), rhs))
    return true;
  return false;
}


template <class T, class F> inline __host__ 
bool operator==(const CudaInterval<T>* lhs, const CudaBox<F>& rhs)
{
  if (std::equal(rhs.elements.begin(), rhs.elements.end(), lhs))
    return true;
  return false;
}


template <class T, class F> inline __host__ 
bool operator==(const CudaBox<T>& lhs, const CudaBox<F>& rhs)
{
  if(lhs.lins() == rhs.lins() && lhs.cols() == rhs.cols())
    if (std::equal( lhs.elements.begin(), lhs.elements.end(), rhs.data()  ))
      return true;
  return false;
}


template <class T, class F> inline __host__ 
bool operator!=(const CudaBox<T>& lhs, const F& rhs)
{
  return !(lhs == rhs);
}

template <class T, class F> inline __host__ 
bool operator!=(const F& lhs, const CudaBox<T>& rhs)
{
  return !(lhs == rhs);
}


// Assignments

// Inference on the size of the array by the length of the vector
template <class T>
template <class F> inline __host__ 
CudaBox<T>& CudaBox<T>::operator=(const CudaInterval<F> *rhs)
{
 try {
    elements.assign(rhs, rhs + this->length());
  }
  catch(const std::exception& e) {
    throw out_of_range("Array index out of range in CudaBox assignment.");
  }
  return *this;
}


template <class T>
template <class F> inline __host__ 
CudaBox<T>& CudaBox<T>::operator=(const CudaBox<F> &rhs)
{
    nl = rhs.nl; nc = rhs.nc;
    elements = rhs.elements;
    return *this;
}


} // namespace intlag

#endif



