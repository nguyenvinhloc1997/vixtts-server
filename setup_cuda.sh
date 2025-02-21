#!/bin/bash

export CUDA_HOME=/usr/lib/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Source this script to set up CUDA environment
# Usage: source setup_cuda.sh 