
CC = nvcc
LDFLAGS = -L. -lcudart
CFLAGS = -I../include



#all: gemm_lib
all: gemm_lib test

gemm_lib:
	$(CC) $(CFLAGS) -Xcompiler -fPIC ../src/spd_blas_API.cpp -c
	$(CC) $(CFLAGS) -Xcompiler -fPIC -shared spd_blas_API.o -o libgemm_lib.so ../src/gemm.cu

test:
	$(CC) $(CFLAGS) gemm_test.cpp  -o gemm_test.x $(LDFLAGS) -lgemm_lib

clean:
	rm -f *~ \#*\# *.o *.x
