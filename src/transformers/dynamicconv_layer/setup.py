#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='dynamicconv_cuda',
    version='0.0.2',
    ext_modules=[
        CUDAExtension(
            name='dynamicconv_cuda',
            sources=[
                'dynamicconv_cuda.cpp',
                'dynamicconv_cuda_kernel.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
