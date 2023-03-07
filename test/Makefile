
CC = nvcc
LDFLAGS = -L. -lcudart
CFLAGS = -I../include -g



all: gemm_lib test

gemm_lib:
	$(CC) $(CFLAGS) -Xcompiler -fPIC -shared -o libgemm_lib.so ../src/gemm.cu

test:
	$(CC) $(CFLAGS) gemm_test.cpp  -o gemm_test.x $(LDFLAGS) -lgemm_lib

clean:
	rm -f *~ \#*\# *.o *.x
