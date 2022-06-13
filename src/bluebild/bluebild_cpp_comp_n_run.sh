#!/bin/bash

#SBATCH --mem 0
#SBATCH --exclusive
#SBATCH --time 03:00:00
#SBATCH --partition build

set -e
set +x

export BLUEBILD_GPU=CUDA

WIPE_BUILD_DIR=1
RUN_PYTHON=1
RUN_TESTS=0
RUN_ADVISOR=0
RUN_VTUNE=0
FILTER="" #32
GCC_VERS=9


#for COMPILER in GCC ICC; do
for COMPILER in GCC; do

    echo; echo
    echo "@@@@@ RUNNING WITH $COMPILER"
    echo

    if   [ $COMPILER == "GCC" ]; then
        export CXXFLAGS="-g -DNDEBUG -fPIC -m64 -Ofast -fopenmp -pedantic -ffast-math  -march=skylake-avx512 -mprefer-vector-width=512 -ftree-vectorize -funsafe-math-optimizations -lm"
    elif [ $COMPILER == "ICC" ]; then
        export CXXFLAGS="-g -DNDEBUG -fPIC -m64 -Ofast -qopenmp -qopt-report=2 -qopt-report-phase=vec -xCORE-AVX512 -qopt-zmm-usage=high"
    fi

    module purge
    if [ $BLUEBILD_GPU == "OFF" ]; then
        if [ $COMPILER == "GCC" ]; then
            #module load gcc openblas/0.3.10-openmp cmake fftw
            if [ $GCC_VERS == 8 ]; then
                module load gcc openblas/0.3.10-openmp cmake fftw
            elif [ $GCC_VERS == 9 ]; then
                module load gcc/9.3.0 mvapich2 openblas/0.3.10-openmp cmake fftw
            else
                echo "Fatal. Unknown GCC version $GCC_VERS"
                exit 1
            fi
        elif [ $COMPILER == "ICC" ]; then
            module load intel intel-mkl cmake fftw
        else
            echo "Fatal: unknown compiler $COMPILER. Only knows GCC and ICC"
            exit 1
        fi
    else # assumes cuda for now
        if [ $COMPILER == "GCC" ]; then
            if [ $GCC_VERS == 8 ]; then
                module load gcc cuda/11.0 openblas/0.3.10-openmp cmake fftw
            elif [ $GCC_VERS == 9 ]; then
                echo "gcc9"
                module load gcc/9.3.0-cuda mvapich2 cuda/11.0 openblas/0.3.10-openmp cmake fftw
            else
                echo "Fatal. Unknown GCC version $GCC_VERS"
                exit 1
            fi
        elif [ $COMPILER == "ICC" ]; then
            module load intel cuda/11.0 intel-mkl cmake fftw
        else
            echo "Fatal: unknown compiler $COMPILER. Only knows GCC and ICC"
            exit 1
        fi 
    fi
    module list

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate pype-111
    echo "CONDA_PREFIX = $CONDA_PREFIX"

    pip show bluebild || echo "bluebild package not found (expected :-))"


    export PROJECT_DIR=~/SKA/epfl-radio-astro/
    export FINUFFT_ROOT=$PROJECT_DIR/finufft
    [ -d $FINUFFT_ROOT ] || (echo "FINUFFT_ROOT >>$FINUFFT_ROOT<< not found" && exit 1)
    export CUFINUFFT_ROOT=$PROJECT_DIR/cufinufft
    [ -d $CUFINUFFT_ROOT ] || (echo "CUFINUFFT_ROOT >>$CUFINUFFT_ROOT<< not found" && exit 1)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FINUFFT_ROOT/lib:$CUFINUFFT_ROOT/lib

    #CMAKE_BUILD_DIR=.
    CMAKE_BUILD_DIR=build_$COMPILER
    
    # Watch out!
    if [ $WIPE_BUILD_DIR == 1 ]; then
        echo "Wiping off ./$CMAKE_BUILD_DIR..."
        [ -d ./$CMAKE_BUILD_DIR ] && rm -rf ./$CMAKE_BUILD_DIR
    fi

    if module is-loaded intel; then
        export LD_PRELOAD=${LD_PRELOAD}:$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_intel_lp64.so:$MKLROOT/lib/intel64/libmkl_intel_thread.so
        export LD_PRELOAD=${LD_PRELOAD}:$INTEL_MKL_ROOT/lib/intel64/libiomp5.so
    fi

    if [ 1 == 1 ]; then
        cmake -S. -B$CMAKE_BUILD_DIR -DBLUEBILD_GPU=$BLUEBILD_GPU -DCMAKE_BUILD_TYPE="BB_CUSTOM" -DMARLA_ROOT="~/SKA/epfl-radio-astro/marla_gf"
        cmake --build $CMAKE_BUILD_DIR -- VERBOSE=1
    else
        if [ 1 == 1 ]; then
            declare -a files_to_del=(
                "./Makefile" "./cmake_install.cmake" "./BLUEBILD*.cmake" "./CMakeCache.txt"
                "./python/Makefile" "./python/cmake_install.cmake"
                "./src/Makefile" "./src/cmake_install.cmake"
                "./tests/Makefile" "./tests/cmake_install.cmake")
            for todel in "${files_to_del[@]}"; do
                [ -f $todel ] && rm $todel 
            done
            declare -a dirs_to_del=(
                "./CMakeFiles" "./tests/CMakeFiles/" "./python/CMakeFiles" "./_deps")
            for todel in "${dirs_to_del[@]}"; do
                [ -d $todel ] && rm -rf $todel 
            done
            exit 0
        fi
        cmake .
        make
    fi

    if [ 1 == 0 ]; then
        echo; echo;
        echo "### Finding bluebild library"
        echo
        echo "@@@ libbluebild.so in $CONDA_PREFIX:"
        find $CONDA_PREFIX -name libbluebild.so -printf "%TY-%Tm-%Td %TH:%TM:%.2TS  %p\n"
        echo;
        echo "@@@ libbluebild.so in pypeline:"
        find ../..         -name libbluebild.so -printf "%TY-%Tm-%Td %TH:%TM:%.2TS  %p\n"
        echo
        if [ 1 == 0 ]; then
            echo "### Checking bluebild.egg-link"
            echo
            BB_EGGLINK=`find $CONDA_PREFIX -name bluebild.egg-link`
            echo "BB_EGGLINK = $BB_EGGLINK"
            ls -l $BB_EGGLINK
            
            exit 0
            
            SRC_PATH=`sed -n '1p' $BB_EGGLINK`
            echo "SRC_PATH = $SRC_PATH"
            find $SRC_PATH -name libbluebild.so -printf "%TY-%Tm-%Td %TH:%TM:%.2TS  %p\n"
            OLD=`find $SRC_PATH -name libbluebild.so | sed -n '1p'`
            echo "OLD = $OLD"
            ls -l $OLD
            NEW=./$CMAKE_BUILD_DIR/src/libbluebild.so
            echo "NEW = $NEW"
            ls -l $NEW
            
            #cp -pv $NEW $OLD
        fi
    fi

    echo
    PYTHON=`which python`
    echo PYTHON=$PYTHON
    $PYTHON -V
    echo

    echo "@@@ Running python import bluebild"
    export PYTHONPATH=/home/orliac/SKA/epfl-radio-astro/pypeline/src/bluebild/build_GCC/python/
    cd ../..
    $PYTHON -c "import bluebild"
    #strace -o trace_output.txt $PYTHON -c "import bluebild; print(bluebild.__file__)" #ctx = bluebild.Context(bluebild.ProcessingUnit.AUTO)"
    cd -


    PY_SCRIPT=../../examples/simulation/lofar_bootes_ss_cpp.py

    export MKL_VERBOSE=0
    export OMP_DISPLAY_AFFINITY=0
    export NUMEXPR_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    if [ $RUN_PYTHON == 1 ]; then
        FIRST=1
        for NTHREADS in 1 #2 4 8 16 32 36
        #for NTHREADS in 36
        do
            export OMP_NUM_THREADS=$NTHREADS
            #time $PYTHON $PY_SCRIPT
            $PYTHON $PY_SCRIPT
            if [ $FIRST == 1 ]; then
                cp -v ./lofar_ss_32.json ./tests/data
                cp -v ./lofar_ss_64.json ./tests/data
                FIRST=0
            fi
        done
    fi


    TEST_SS=$CMAKE_BUILD_DIR/tests/run_tests

    if [ $RUN_TESTS == 1 ]; then
        OMP_NUM_THREADS=40 $TEST_SS --gtest_filter=*$FILTER
        #exit 0
        echo
    fi

    # Output directory for Intel profiling
    SCRATCH=/scratch/$USER
    regex="^i[:digit:][:digit]$"
    myhost=`hostname`
    if [[ "$myhost" == "izar" || $myhost =~ [^i:digit::digit:$] ]]; then
        SCRATCH=/scratch/izar/$USER
    fi
    TEST_DIR=$SCRATCH/css-cpp/test01/$COMPILER
    [ -d $TEST_DIR ] && rm -r $TEST_DIR
    mkdir -p $TEST_DIR
    echo TEST_DIR = $TEST_DIR
    echo

    if [ $RUN_ADVISOR == "1" ]; then
        source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-advisor-2021.4.0-any7cfov5s4ujprr7plf7ks7xzoyqljz/setvars.sh --force || echo "ignoring warning"
        #advixe-cl --collect=roofline --project-dir=$TEST_DIR/advisor -- $PYTHON $PY_SCRIPT
        advixe-cl --collect=roofline --project-dir=$TEST_DIR/advisor -search-dir src:=./src -- $TEST_SS --gtest_filter=*$FILTER
    fi

    if [ $RUN_VTUNE == "1" ]; then
        echo "### Intel VTune Amplifier"
        source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh || echo "ignoring warning"
        which vtune
        vtune -collect hotspots                                -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_hs  -- $PYTHON $PY_SCRIPT
        vtune -collect hpc-performance    -call-stack-mode all -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_hpc -- $PYTHON $PY_SCRIPT
        #vtune -collect memory-consumption -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_mem -- $PYTHON $PY_SCRIPT

        #vtune-gui $TEST_DIR/vtune_hpc/vtune_hpc.vtune&
    fi
    echo; echo

done
