#include "spd_blas.hpp"

/*
template<typename T>
void spd_gpu_gemm(size_t M, size_t N, size_t K,
		  T alpha,
		  T *A, size_t lda, T *B, size_t ldb,
		  T beta,
		  T *C, size_t ldc)
{
  _gemm_ker<<<1, 128, 0, streams[cur_stream]>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  cur_stream = (cur_stream + 1) % SPD_NUM_STREAMS;

  cout<<cur_stream<<endl;
}
*/


int main(){

  double *a, *b, *c, *gpu_c;
  double *dev_a, *dev_b, *dev_c;

  int N = 512;

  //  init_spd();
  
  posix_memalign((void**)&a, 64, sizeof(double)*N*N);
  posix_memalign((void**)&b, 64, sizeof(double)*N*N);
  posix_memalign((void**)&c, 64, sizeof(double)*N*N);

  posix_memalign((void**)&gpu_c, 64, sizeof(double)*N*N);  
  
  cudaMalloc((void**) &dev_a, N * N * sizeof(double));
  cudaMalloc((void**) &dev_b, N * N * sizeof(double));  
  cudaMalloc((void**) &dev_c, N * N * sizeof(double));

  cudaMemcpy(dev_a, a, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();


  int lda, ldb, ldc;
  lda = ldb = ldc = N;
  double *A = dev_a;
  double *B = dev_b;
  double *C = dev_c;
  double alpha = 1.0;
  double beta = 1.0;  
  
  for (int i = 0; i != 4; ++i){   
    
    
    spd_gemm(N/4, N, N,
	   1.0,
	   dev_a + i*N*N/4, N, dev_b, N,
	   1.0,
	   dev_c + i*N*N/4, N,
	   SPD_GPU);
   
  }

  cudaDeviceSynchronize();

  
  spd_gemm(N, N, N,
	   1.0,
	   dev_a, N, dev_b, N,
	   1.0,
	   dev_c, N,
	   SPD_GPU);
  

  cudaDeviceSynchronize();

  cudaMemcpy(gpu_c, dev_c, N * N * sizeof(double),
	     cudaMemcpyDeviceToHost);  

  cout<<"Success"<<endl;

  
  //  finalize_spd();
  
  free(a);
  free(b);
  free(c);

  free(gpu_c);  

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
