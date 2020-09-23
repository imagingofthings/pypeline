source  ~software/source_spack.sh
module load intel intel-mkl
export LC_ALL=en_US
make

#gcc zgemm-splat.o -shared -o zgemm-splat.so
#gcc zgemm-splat.o -shared -o zgemm-splat.so