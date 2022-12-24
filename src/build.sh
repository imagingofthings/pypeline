# ssh -XY etolley@test-u25-n01
source  ~software/source_spack.sh
module load intel intel-mkl
module load python/3.7.7
export LC_ALL=en_US
make clean
make

#gcc zgemm-splat.o -shared -o zgemm-splat.so
#gcc zgemm-splat.o -shared -o zgemm-splat.so