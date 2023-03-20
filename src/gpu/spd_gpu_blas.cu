#include <cuda_runtime.h>
#include <iostream>
#include "spd_gpu_blas.hpp"

using namespace std;

 cudaStream_t streams[4];

 int cur_stream = 0;
 bool initialized = false;

void init_spd()
{
  if (!initialized)
    {
      for (int i = 0; i < SPD_NUM_STREAMS; ++i){
	cudaError_t err = cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
	if (err != cudaSuccess)
	  cout<<"Err create stream"<<endl;
      }

      initialized = true;
    }
}


void finalize_spd()
{
  if (initialized)
    {
      for (int i = 0; i < SPD_NUM_STREAMS; ++i)
	cudaStreamDestroy(streams[i]);
      
      cudaDeviceReset();

      initialized = false;
    }
}

