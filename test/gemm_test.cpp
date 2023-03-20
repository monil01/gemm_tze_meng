#include <iostream>
#include "spd_blas.hpp"
#include <cuda_runtime.h>
#include <math.h>

using namespace std;


int main()
{
  double *a, *b, *c, *gpu_c;
  double *dev_a, *dev_b, *dev_c;

  int N = 256;

  posix_memalign((void**)&a, 64, sizeof(double)*N*N);
  posix_memalign((void**)&b, 64, sizeof(double)*N*N);
  posix_memalign((void**)&c, 64, sizeof(double)*N*N);

  posix_memalign((void**)&gpu_c, 64, sizeof(double)*N*N);  

  for (int i = 0; i != N * N; ++i)
    {
      a[i] = 1;
      b[i] = 1;
      c[i] = 1;
    }

  cudaMalloc((void**) &dev_a, N * N * sizeof(double));
  cudaMalloc((void**) &dev_b, N * N * sizeof(double));  
  cudaMalloc((void**) &dev_c, N * N * sizeof(double));


  cudaMemcpy(dev_a, a, N * N * sizeof(double),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * N * sizeof(double),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, N * N * sizeof(double),
	     cudaMemcpyHostToDevice);

  
  spd_gemm(N, N, N,
	   1.0,
	   dev_a, N, dev_b, N,
	   1.0,
	   dev_c, N,
	   SPD_GPU);

  spd_gemm(N, N, N,
	     1.0,
	     a, N, b, N,
	     1.0,
	     c, N,
	     SPD_CPU);
  
  cudaDeviceSynchronize();
  
  cudaMemcpy(gpu_c, dev_c, N * N * sizeof(double),
	     cudaMemcpyDeviceToHost);

  bool correct = true;
  for (int i = 0; i != N*N; ++i)
    correct &= (fabs(gpu_c[i] - c[i]) < 1e-15);  

  cout<<(correct ? "Yes" : "No")<<endl;
  
  free(a);
  free(b);
  free(c);

  free(gpu_c);  

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
