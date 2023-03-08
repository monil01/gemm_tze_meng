#pragma once

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#ifndef __GPU_BLAS_STATIC__
#define __GPU_BLAS_STATIC__

static int SPD_NUM_STREAMS= 4;

void init_spd();
void finalize_spd();

template<typename T>
__global__ void _gemm_ker(size_t M, size_t N, size_t K,
			  T alpha,
			  T *A, size_t lda, T *B, size_t ldb,
			  T beta,
		 	  T *C, size_t ldc);

template<typename T>
void spd_gpu_gemm(size_t M, size_t N, size_t K,
		  T alpha,
		  T *A, size_t lda, T *B, size_t ldb,
		  T beta,
		  T *C, size_t ldc);

#endif


