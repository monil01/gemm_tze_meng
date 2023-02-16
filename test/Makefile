
CC = nvcc
LDFLAGS = -lcudart
CFLAGS = -I../include



all: gemm_lib test

gemm_lib:
	$(CC) $(CFLAGS) ../src/gemm.cu -c

test:
	$(CC) $(CFLAGS) gemm.o gemm_test.cpp -o gemm_test.x $(LDFLAGS)

clean:
	rm -f *~ \#*\# *.o *.x
