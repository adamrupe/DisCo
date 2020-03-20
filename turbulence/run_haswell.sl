#!/bin/bash -l

#SBATCH -p regular
#SBATCH -t 0:15:00
#SBATCH --nodes=59
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH -C haswell
#SBATCH -J sq-vort
#SBATCH -o $SCRATCH/results/turbulence/SC19/result-18/%j_hsl.log
#SBATCH -e $SCRATCH/results/turbulence/SC19/result-18/%j_hsl.err
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

srun -n 118 python -m tbb -p 32 --ipc ./turb.py >> $SCRATCH/results/turbulence/SC19/result-18/params.txt

# for interactive -- see comment top of turb.py to calc -N and -n from desired lightcone params and work size:
# salloc -N 59 -C haswell -q interactive -t 01:00:00
# srun -n 118 --tasks-per-node=2 --cpus-per-task=32 python -m tbb -p 32 --ipc ./turb.py >> $SCRATCH/results/turbulence/SC19/result-21/params.txt
