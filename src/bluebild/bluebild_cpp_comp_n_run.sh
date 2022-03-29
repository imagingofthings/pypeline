#!/bin/bash

#SBATCH --mem 0
#SBATCH --exclusive
#SBATCH --time 03:00:00

set -e
set +x

export BLUEBILD_GPU=OFF
export MARLA_ROOT="~/SKA/epfl-radio-astro/marla_gf"

WIPE_BUILD=0
RUN_PYTHON=1
RUN_TESTS=0
RUN_ADVISOR=0
RUN_VTUNE=0
FILTER=32

for COMPILER in GCC ICC; do
#for COMPILER in ICC; do

    echo; echo
    echo "@@@@@ RUNNING WITH $COMPILER"
    echo

    # Set this one to disable autoset of default build type in CMake
    export CMAKE_BUILD_TYPE="ANYTHING" 

    if   [ $COMPILER == "GCC" ]; then
        export CXXFLAGS="-g -DNDEBUG -fPIC -m64 -Ofast -fopenmp -pedantic -ffast-math  -march=skylake-avx512 -mprefer-vector-width=512 -ftree-vectorize -funsafe-math-optimizations -lm"
    elif [ $COMPILER == "ICC" ]; then
        export CXXFLAGS="-g -DNDEBUG -fPIC -m64 -Ofast -qopenmp -qopt-report=2 -qopt-report-phase=vec -xCORE-AVX512 -qopt-zmm-usage=high"
    fi

    module purge
    if [ $BLUEBILD_GPU == "OFF" ]; then
        if [ $COMPILER == "GCC" ]; then
            #module load gcc openblas/0.3.10-openmp cmake fftw
            module load gcc openblas cmake fftw
        elif [ $COMPILER == "ICC" ]; then
            module load intel intel-mkl cmake fftw
        else
            echo "Fatal: unknown compiler $COMPILER. Only knows GCC and ICC"
            exit 1
        fi
    else # assumes cuda for now
        if [ $COMPILER == "GCC" ]; then
            module load gcc cuda/11.0 openblas cmake fftw
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

    pip show bluebild


    export PROJECT_DIR=~/SKA/epfl-radio-astro/
    export FINUFFT_ROOT=$PROJECT_DIR/finufft
    [ -d $FINUFFT_ROOT ] || (echo "FINUFFT_ROOT >>$FINUFFT_ROOT<< not found" && exit 1)
    export CUFINUFFT_ROOT=$PROJECT_DIR/cufinufft
    [ -d $CUFINUFFT_ROOT ] || (echo "CUFINUFFT_ROOT >>$CUFINUFFT_ROOT<< not found" && exit 1)
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FINUFFT_ROOT/lib:$CUFINUFFT_ROOT/lib

    CMAKE_BUILD_DIR=build_$COMPILER

    # Watch out
    if [ $WIPE_BUILD == 1 ]; then
        echo ./$CMAKE_BUILD_DIR
        [ -d ./$CMAKE_BUILD_DIR ] && rm -rf ./$CMAKE_BUILD_DIR
    fi

    if module is-loaded intel; then
        export LD_PRELOAD=${LD_PRELOAD}:$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_intel_lp64.so:$MKLROOT/lib/intel64/libmkl_intel_thread.so
        export LD_PRELOAD=${LD_PRELOAD}:$INTEL_MKL_ROOT/lib/intel64/libiomp5.so
        #echo "LD_PRELOAD = $LD_PRELOAD"
    else
        #export LD_LIBRARY_PATH=$OPENBLAS_ROOT/lib/libopenblas.so:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64/libpthread.so.0
        #echo "No export for GCC"
        #export INTEL_ROOT=/ssoft/spack/arvine/v1/opt/spack/linux-rhel7-haswell/gcc-4.8.5/intel-19.0.5-wvfmqbnhon3gjhmctxz4bfs6z7udbgix
        #export LD_LIBRARY_PATH=$INTEL_ROOT/lib/intel64/:$LD_LIBRARY_PATH
        #export LD_PRELOAD=$INTEL_ROOT/lib/intel64/libintlc.so.5:$INTEL_ROOT/lib/intel64/libsvml.so
    fi

    cmake -S. -B$CMAKE_BUILD_DIR
    cmake --build $CMAKE_BUILD_DIR -- VERBOSE=1

    echo
    PYTHON=`which python`
    echo PYTHON=$PYTHON
    $PYTHON -V
    echo

    PY_SCRIPT=../../examples/simulation/lofar_bootes_ss_cpp.py

    export MKL_VERBOSE=0
    export OMP_DISPLAY_AFFINITY=0
    export NUMEXPR_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    if [ $RUN_PYTHON == 1 ]; then
        FIRST=1
        for NTHREADS in 1 #2 4 8 16 32 36
        do
            export OMP_NUM_THREADS=$NTHREADS
            #time $PYTHON $PY_SCRIPT
            $PYTHON $PY_SCRIPT
            if [ $FIRST == 1 ]; then
                cp -v ./lofar_ss_32.json ./tests/data
                cp -v ./lofar_ss_64.json ./tests/data
                FIRST=2
            fi
        done
    fi


    TEST_SS=$CMAKE_BUILD_DIR/tests/run_ss_tests

    if [ $RUN_TESTS == 1 ]; then
        $TEST_SS --gtest_filter=*$FILTER
        #exit 0
        echo
    fi

    # Output directory for Intel profiling
    SCRATCH=/scratch/$USER
    [ `hostname` == 'izar' ] && SCRATCH=/scratch/izar/$USER
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
