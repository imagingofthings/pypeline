#!/bin/bash -l
#SBATCH --job-name=job_name
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=c31

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
module load daint-gpu                   # (or daint-mc)
module load cray-python/3.8.5.0
module load Boost/1.70.0-CrayGNU-20.11-python3
module unload gcc/9.3.0
module load gcc/10.2.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314

conda activate pypeline
python analysis_bluebild.py
