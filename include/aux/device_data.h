//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef DEVICE_DATA_H
#define DEVICE_DATA_H

#include <stdexcept>
#include "cuda_error.h"


namespace intlag {

template <class T>
class DeviceData {

  public:

		DeviceData() : size_(0), data_(0) { }

		DeviceData(int n) : size_(n*sizeof(T)) {
			CHECKED_CALL( cudaMalloc((void**) &data_, size_) );
		}

		DeviceData(int n, const T* x) : size_(n*sizeof(T)) {
			CHECKED_CALL( cudaMalloc((void**) &data_, size_) );
			CHECKED_CALL( cudaMemcpy(data_, x, size_, cudaMemcpyHostToDevice) );
		}

		~DeviceData() {
			if (data_) CHECKED_CALL( cudaFree(data_) );
		}

		void toHost(T* x) const {
			CHECKED_CALL( cudaMemcpy(x, data_, size_, cudaMemcpyDeviceToHost) );
		}

		void toHost(T* x, int n) const {
      //assert( n * sizeof(T) <= size_ );
			CHECKED_CALL( cudaMemcpy(x, data_, n*sizeof(T), cudaMemcpyDeviceToHost) );
		}

		T const* data() const {
			return data_;
		}

		T* data() {
			return data_;
		}

		DeviceData& copyAndDestroy(DeviceData& other) {
    	if (this != &other) {
				size_ = other.size_;
				data_ = other.data_;
				other.data_ = NULL;
			}

			return *this;
    }


	private:
		T* data_;
		size_t size_;
};

template<class T>
void swapByReference(DeviceData<T>& a, DeviceData<T>& b){

	DeviceData<T> aux;
	aux.copyAndDestroy(a);
	a.copyAndDestroy(b);
	b.copyAndDestroy(aux);
}

} // namespace intlag

#endif
