#!/bin/bash -l

#SBATCH -p regular
##SBATCH --reservation=DisCoHackathon2
#SBATCH -t 0:30:00
#SBATCH --nodes=36
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH -C haswell
#SBATCH -J disco_ws
#SBATCH -o $SCRATCH/results/Jupiter/SC19/result-3/%j_hsl.log
#SBATCH -e $SCRATCH/results/Jupiter/SC19/result-3/%j_hsl.err
#SBATCH --mail-user=atrupe@ucdavis.edu
#SBATCH --mail-type=END

module load python/3.6-anaconda-4.4
module unload darshan

export MKL_NUM_THREADS=16
export NUMBA_THREADING_LAYER=TBB
export KMP_BLOCKTIME=10

source /global/common/software/ProjectDisCo/pyenv/kmeans_hsw/bin/activate

#parallel
# n = total no. of processes
# tasks-per-node = mpi processes per node

srun -n 72 python -m tbb -p 32 --ipc ./jupiter.py >> $SCRATCH/results/Jupiter/SC19/result-3/params.txt

# for interactive:
# salloc -N 36 -C haswell -q interactive -t 01:00:00
# srun -n 72 --tasks-per-node=2 --cpus-per-task=32 python -m tbb -p 32 --ipc ./jupiter.py >> $SCRATCH/results/Jupiter/SC19/result-3/params.txt
