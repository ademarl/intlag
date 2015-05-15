
#ifndef CUDA_BOX_H
#define CUDA_BOX_H

#include <stdexcept>
#include <vector>
#include "box_interval.h"
#include "cuda_interval_lib.h"
using namespace std;


template <class T>
class box_gpu {

  public:

		box_gpu () {}

		~box_gpu () { }

    box_gpu (int n) {
      typename vector< interval_gpu<T> >::iterator it = elements.begin();
      elements.insert(it, n, interval_gpu<T>());
    }

    box_gpu (interval_gpu<T> a, int n) {
      typename vector< interval_gpu<T> >::iterator it = elements.begin();
      elements.insert(it, n, a);
    }

    box_gpu (const interval_gpu<T>* a, int n) {
			elements.assign(a, a+n);
    }

    box_gpu(const box_gpu<T> &x) {
      elements = x.elements;
    }

		interval_gpu<T>const* data() const {
			return &elements[0];
		}

		interval_gpu<T>* data() {
			return &elements[0];
		}


    interval_gpu<T> at (int n) {
      if (n >= (int)elements.size() || n < 0)
        throw out_of_range("Box index out of range.");
      return elements.at(n);
    }

		interval_gpu<T> operator[] (int n) {
			return elements[n];
		}

    int length() const {
      return elements.size();
    }

		bool empty() {
			return elements.empty();
		}

	private:
		std::vector< interval_gpu<T> > elements;

};

#endif
