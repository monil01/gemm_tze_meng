#include <cstddef>
#include "spd_blas.hpp"
#include "spd_blas_API.hpp"


void spd_gemm_double(size_t M, size_t N, size_t K,
	      double alpha,
	      double *A, size_t lda, double *B, size_t ldb,
	      double beta,
	      double *C, size_t ldc,
	      SPD_DEVICE device_type)
{

    spd_gemm(M, N, K,
             alpha,
             A, lda, B, ldb,
             beta,
             C, ldc,
             device_type);

}

void spd_gemm_float(size_t M, size_t N, size_t K,
	      float alpha,
	      float *A, size_t lda, float *B, size_t ldb,
	      float beta,
	      float *C, size_t ldc,
	      SPD_DEVICE device_type)
{
    spd_gemm(M, N, K,
             alpha,
             A, lda, B, ldb,
             beta,
             C, ldc,
             device_type);

}

