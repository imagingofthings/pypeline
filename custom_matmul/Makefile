CC = icc
#LIBS = -L/home/foureste/Projects/OpenBLAS-0.2.14/ -lopenblas 
LIBS = -mkl
INCLUDES =
CFLAGS = -Ofast -qopt-report=2 -qopt-report-phase=vec -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp -g -I./

all:	dgemm-splat zgemm-splat 


dgemm-splat: dgemm.o dgemm-splat.o
	$(CC) -o $@ dgemm.o dgemm-splat.o $(CFLAGS) $(LIBS) $(INCLUDES)

zgemm-splat: zgemm.o zgemm-splat.o
	$(CC) -o $@ zgemm.o zgemm-splat.o $(CFLAGS) $(LIBS) $(INCLUDES)

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $<

clean:
	rm dgemm-splat zgemm-splat *.o 

