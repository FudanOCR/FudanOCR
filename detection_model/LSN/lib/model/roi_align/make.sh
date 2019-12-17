#!/usr/bin/env bash

export CUDA_PATH=/usr/local/cuda/
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-9.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
