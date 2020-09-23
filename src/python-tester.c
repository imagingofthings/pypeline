#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <float.h>
#include <math.h>
#include <complex.h>

void test_c_int( const int i){
	printf("Here is the c_int received by python_tester: %d\n", i);
}

void test_c_complex(  double complex c){
	printf("Here is the c_complex received by python_tester: %f + i%f\n", creal(c), cimag(c));
}

void test_c_complex_fromPython(  double complex * c){
	test_c_complex(c[0]);
}

void test_c_complex_pointer(  double complex * c){
	printf("Here is the first element of * c_complex received by python_tester: %f + i%f\n", creal(c[0]), cimag(c[0]));
}

int main (int argc, char *argv[]){
	return 0;
}