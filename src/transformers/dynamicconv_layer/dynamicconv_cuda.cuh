/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define SHFL_MASK 0xffffffff

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void dynamicconv_forward_kernel(const scalar_t* input,
                                const scalar_t* weight,
                                int minibatch, 
                                int sequenceLength,
                                int numFeatures, 
                                int numFiltersInBlock,
                                int numHeads,
                                scalar_t* output);

template<int FS, int SB, int padding_l, typename scalar_t>
__global__
void dynamicconv_backward_kernel(
    const scalar_t* gradOutput, // B * C * T
    const scalar_t* input, // B * C * T
    const scalar_t* weight,
    int minibatch,
    int sequenceLength,
    int numFeatures,
    int numFiltersInBlock,
    int numHeads,
    scalar_t* gradWeight,
    scalar_t* gradInput); // B * H * k * T
