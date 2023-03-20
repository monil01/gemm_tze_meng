#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>

#define CUDA 1

#if HIP
#include <hip/hip_runtime.h>
#include <hipblas.h>
#elif CUDA
// #include <hip/hip_runtime.h>
// #include <hipblas.h>
#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipBlockIdx_x  blockIdx.x
#define hipBlockIdx_y  blockIdx.y
#define hipBlockDim_x blockDim.x
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipSuccess cudaSuccess
#endif

//#include "gemm.hpp"


#include <iostream>
#include "spd_gpu_blas.hpp"

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

#define HIPBLAS_ASSERT(x) (assert((x)==HIPBLAS_STATUS_SUCCESS))

#define REAL 0
#define IMAG 1
#define COMPLEX 2

#define VEGA64 0
#define V100   1

#define ACC_T double

#define DEFAULT_FMA(c, a, b) c += a * b;

template<typename T>
__device__ void _fma(double &c, const T &a, const T &b) {
    DEFAULT_FMA(c, a, b);
}

#define DEF_AB() \
    acc_t ab[NT][MT]; \
    _Pragma("unroll") \
    for (int j = 0; j < NT; j++) { \
        _Pragma("unroll") \
        for (int i = 0; i < MT; i++) { \
            ab[j][i] = {}; \
        } \
    }

// Not sure I can get the rounding thing to work..?
#define DEF_LOAD() \
a_vector_t Al[(MR*KR + (T*AVEC) - 1)/(T*AVEC)]; \
b_vector_t Bl[(NR*KR + (T*BVEC) - 1)/(T*BVEC)];


    // for (int ppl = 0; ppl < KR/(T/(MR/AVEC)); ppl++) {


#define LOAD_A(p) { \
    static_assert(T % (MR/AVEC) == 0); \
    int il = (tid % (MR/AVEC)); \
    int pl = (tid / (MR/AVEC)); \
    if (pl < KR) { \
        _Pragma("unroll") \
        for (int ppl = 0; ppl < KR/(T/(MR/AVEC)); ppl++) { \
            Al[ppl] = *((a_vector_t *) (A + ((p) + ppl*(T/(MR/AVEC)) + pl) * m + ib * MR + il*AVEC)); \
        } \
    } \
}

#define LOAD_B(p) { \
    static_assert(T % (NR/BVEC) == 0); \
    int jl = (tid % (NR/BVEC)); \
    int pl = (tid / (NR/BVEC)); \
    if (pl < KR) { \
        _Pragma("unroll") \
        for (int pplr = 0, pplm = 0; pplm < KR; pplr++, pplm += (T/(NR/BVEC))) { \
            Bl[pplr] = *((b_vector_t *) (B + ((p) + pplm + pl) * n + jb * NR + jl*BVEC)); \
        } \
    } \
}

#define STORE_A(buffer_id) { \
    int il = (tid % (MR/AVEC)); \
    int pl = (tid / (MR/AVEC)); \
    if (pl < KR) { \
        _Pragma("unroll") \
        for (int ppl = 0; ppl < KR/(T/(MR/AVEC)); ppl++) { \
            *((a_vector_t *) (As + (buffer_id)*MR*KR + (ppl*(T/(MR/AVEC)) + pl) * MR + il*AVEC)) = Al[ppl]; \
        } \
    } \
}

#define STORE_B(buffer_id) { \
    int jl = (tid % (NR/BVEC)); \
    int pl = (tid / (NR/BVEC)); \
    if (pl < KR) { \
        _Pragma("unroll") \
        for (int pplr = 0, pplm = 0; pplm < KR; pplr++, pplm += (T/(NR/BVEC))) { \
            *((b_vector_t *) (Bs + (buffer_id)*NR*KR + (pplm + pl) * NR + jl*BVEC)) = Bl[pplr]; \
        } \
    } \
}

#define LOAD(iter_offset) LOAD_A(iter_offset); LOAD_B(iter_offset);
#define STORE(buffer_id) STORE_A(buffer_id); STORE_B(buffer_id);

#define COMPUTE(buffer_id) { \
    for (int pp = 0; pp < KT; pp++) { \
        input_t a[MT], b[NT]; \
        for (int i = 0; i < MT; i++) { \
            a[i] = *(As + (buffer_id) * MR*KR + (pp) * MR + i * MB*MW + iw*MW + it); \
        } \
        for (int j = 0; j < NT; j++) { \
            b[j] = *(Bs + (buffer_id) * NR*KR + (pp) * NR + j * NB*NW + jw*NW + jt); \
        } \
        for (int j = 0; j < NT; j++) { \
            for (int i = 0; i < MT; i++) { \
                _fma(ab[j][i], a[i], b[j]); \
            } \
        } \
    } \
}

#define MAX_THREADS_PER_BLOCK T
#define MIN_WARPS_PER_EU 1



extern int cur_stream;
extern cudaStream_t streams[4];

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

/*
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
*/


template<
    size_t MT, size_t NT, size_t KT,
    size_t MW, size_t NW,
    size_t MB, size_t NB,
    typename a_vector_t, typename b_vector_t,
    typename input_t, typename acc_t
>
__global__
void
__launch_bounds__(MW * NW * MB * NB, MIN_WARPS_PER_EU)
gemm_kernel(
        size_t m,
        size_t n,
        size_t k,
	input_t alpha, 
        const input_t * __restrict__ A,
        const input_t * __restrict__ B,
	input_t beta, 
              input_t * __restrict__ C
	
) {
    constexpr int MR = MT * MB * MW;
    constexpr int NR = NT * NB * NW;
    constexpr int KR = KT;
    constexpr int T  = MW * NW * MB * NB;
    constexpr int AVEC = sizeof(a_vector_t) / sizeof(input_t);
    constexpr int BVEC = sizeof(b_vector_t) / sizeof(input_t);

    // block index in grid.
    int ib = hipBlockIdx_x;
    int jb = hipBlockIdx_y;
    // warp index in block.
    int jw = (hipThreadIdx_y % NB);
    int iw = (hipThreadIdx_y / NB);
    // thread index in warp.
    int jt = (hipThreadIdx_x % NW);
    int it = (hipThreadIdx_x / NW);

    int tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;

    __shared__ input_t As[2*MR*KR];
    __shared__ input_t Bs[2*NR*KR];

    DEF_AB();

    // TODO: Software pipeline and vectorize the loading of A and B into reg and storing into shared
    DEF_LOAD();
    // Prologue.
    LOAD(0);
    STORE(0);

    for (int p = 0; p < k - 2*KR; p += 2*KR) {
        // iter 0
        LOAD(p + 1*KR);
        __syncthreads();
        COMPUTE(0);
        STORE(1);
        // iter 1
        LOAD(p + 2*KR);
        __syncthreads();
        COMPUTE(1);
        STORE(0);
    }
    
    // Epilogue.
    LOAD(k - KR);
    __syncthreads();
    COMPUTE(0);
    STORE(1);
    // no load.
    __syncthreads();
    COMPUTE(1);
    // no store.

    _Pragma("unroll")
    for (int j = 0; j < NT; j++) {
        _Pragma("unroll")
        for (int i = 0; i < MT; i++) {
            *((acc_t *) (C +
                (ib * MR + i * MB*MW + iw*MW + it) * n +
                (jb * NR + j * NB*NW + jw*NW + jt) * 1
            )) += alpha * ab[j][i];

        }
    }
} // end mm_kernel func.



template<typename T>
void spd_gpu_gemm(size_t M, size_t N, size_t K,
		  T alpha,
		  T *A, size_t lda, T *B, size_t ldb,
		  T beta,
		  T *C, size_t ldc)
{
  init_spd();


    constexpr int MT = 8;
    constexpr int NT = 8;

    constexpr int KT = 8;
    
    constexpr int MW = 2;
    constexpr int NW = 16;
    
    constexpr int MB = 4;
    constexpr int NB = 1;
    
    constexpr int MR = MT * MW * MB;
    constexpr int NR = NT * NW * NB;
    typedef T load_a_t;
    typedef T load_b_t;
    auto kernel = gemm_kernel<MT, NT, KT, MW, NW, MB, NB, load_a_t, load_b_t, T, T>;
    kernel<<<dim3(M/MR, N/NR), dim3(MW * NW, MB * NB), 0, streams[cur_stream]>>>(M, N, K, alpha, A, B, beta, C);
    //    HIP_ASSERT(hipDeviceSynchronize());
      //  _gemm_ker<<<1, 128, 0, streams[cur_stream]>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    cur_stream = (cur_stream + 1) % SPD_NUM_STREAMS;
}


auto spd_gpu_dgemm = spd_gpu_gemm<double>;
//auto spd_gpu_sgemm = spd_gpu_gemm<float>;
