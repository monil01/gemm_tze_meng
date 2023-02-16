#include "spd_blas.hpp"

void spd_dgemm(size_t M, size_t N, size_t K,
	      double alpha,
	      double *A, size_t lda, double *B, size_t ldb,
	      double beta,
	      double *C, size_t ldc,
	      SPD_DEVICE device_type);

void spd_sgemm(size_t M, size_t N, size_t K,
	      float alpha,
	      float *A, size_t lda, float *B, size_t ldb,
	      float beta,
	      float *C, size_t ldc,
	      SPD_DEVICE device_type);

