CC = nvcc
LDFLAGS = -L. -lcudart
CFLAGS = -I../include 


all: gpu_lib test

gpu_blas:
	$(CC) $(CFLAGS) --default-stream per-thread -Xcompiler -fPIC -shared  ../src/gpu/spd_gpu_blas.cu -c 

gpu_gemm:
	$(CC) $(CFLAGS) --default-stream per-thread -Xcompiler -fPIC -shared  ../src/gpu/spd_gpu_gemm.cu -c


gpu_lib: gpu_blas gpu_gemm
	ar cr libspd_gpu.a spd_gpu_blas.o spd_gpu_gemm.o


stream: gpu_lib 
	$(CC) $(CFLAGS) stream.cu -o stream.x $(LDFLAGS) -lspd_gpu 

test:	gpu_lib
	$(CC) $(CFLAGS) gemm_test.cpp  -o gemm_test.x $(LDFLAGS) -lspd_gpu

clean:
	rm -f *~ \#*\# *.o *.x *.so *.a
