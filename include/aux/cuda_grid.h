//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef CUDA_GRID_H
#define CUDA_GRID_H

#define BLOCK_SIZE 32
#define BLOCK_X 8
#define BLOCK_Y 8

namespace intlag {


//template<T> // will grid change between float and double? what about specific functions?
class CudaGrid {
  public :
    static dim3 blocks(int N) {
      return dim3((N+BLOCK_SIZE)/BLOCK_SIZE);
    }

    static dim3 threads() {
      return dim3(BLOCK_SIZE);
    }

    static dim3 blocks2(int N, int M) {
      return dim3((N+BLOCK_X)/BLOCK_X,(M+BLOCK_Y)/BLOCK_Y);
    }

    static dim3 threads2() {
      return dim3(BLOCK_X, BLOCK_Y);
    }
};

} // namespace intlag
#endif
