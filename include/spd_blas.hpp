#include "spd_gpu_blas.hpp"
#include <stdio.h>
#include <iostream>

using namespace std;

#ifndef __SPD_BLAS_HEADER
#define __SPD_BLAS_HEADER


enum SPD_DEVICE {SPD_CPU, SPD_GPU, SPD_AUTO};
enum SPD_DIRECTION {SPD_LEFT, SPD_RIGHT};
enum SPD_UPLO {SPD_LOWER, SPD_UPPER};
enum SPD_DIAG {SPD_UNIT_DIAG, SPD_NONUNIT_DIAG};
enum SPD_TRANS{SPD_TRANSPOSE, SPD_NOTRANSPOSE};

template<typename T>
void spd_gemm(size_t M, size_t N, size_t K,
	      T alpha,
	      T *A, size_t lda, T *B, size_t ldb,
	      T beta,
	      T *C, size_t ldc,
	      SPD_DEVICE device_type)
{
  if (device_type == SPD_CPU)
    {
      for (int i = 0; i != M; ++i)
	for (int j = 0; j != N; ++j)
	  for (int p = 0; p != K; ++p)
	    {
	      C[i*ldc + j] += A[i*lda + p] * B[p*ldb + j];
	    }
    }
  else
    {
      spd_gpu_gemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}


template<typename T>
void spd_trsm(SPD_DIRECTION SIDE,
	      SPD_UPLO UPLO,
	      SPD_TRANS TRANS,
	      SPD_DIAG UNIT_DIAG,
	      int M, int N,
	      T alpha, 
	      T *L, size_t ldl,
	      T *X, size_t ldx,
	      SPD_DEVICE device_type);


//auto spd_gemm_dgemm = spd_gemm<double>;

#endif
