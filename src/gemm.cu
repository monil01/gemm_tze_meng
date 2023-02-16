#include <iostream>
#include "spd_gpu_blas.hpp"

using namespace std;

#define CUDA_CHECK(routine, msg)   \
  { \
    cudaError_t status;            \
    status = (routine);            \
    if (status != cudaSuccess){    \
      cout<<msg<<endl;	           \
      return status; \
    } \
  }
template<typename T>
__global__ void _gemm_ker(size_t M, size_t N, size_t K,
			     T alpha,
			     T *A, size_t lda, T *B, size_t ldb,
			     T beta,
			     T *C, size_t ldc)
{
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      blockIdx.y == 0 && threadIdx.y == 0)
    {
      for (int i = 0; i != M; ++i)
	for (int j = 0; j != N; ++j)
	  for (int p = 0; p != K; ++p)
	    {
	      C[i*ldc + j] += A[i*lda + p] * B[p*ldb + j];
	    }
    }
}


template<typename T>
void spd_gpu_gemm(size_t M, size_t N, size_t K,
		  T alpha,
		  T *A, size_t lda, T *B, size_t ldb,
		  T beta,
		  T *C, size_t ldc)
{
  _gemm_ker<<<4, 128, 0>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

auto spd_gpu_dgemm = spd_gpu_gemm<double>;
auto spd_gpu_sgemm = spd_gpu_gemm<float>;
