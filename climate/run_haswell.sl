#!/bin/bash -l

#SBATCH -p regular
##SBATCH --reservation=DisCoHackathon2
#SBATCH -t 0:30:00
#SBATCH --nodes=8
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH -C haswell
#SBATCH -J disco_ws
#SBATCH -o $SCRATCH/results/climate/SC19/result-1/%j_hsl.log
#SBATCH -e $SCRATCH/results/climate/SC19/result-1/%j_hsl.err
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

# RESDIR="/global/project/projectdirs/ProjectDisCo/scaling_results/kmeans/haswell-weak/results"
#
# echo "SLURM_JOB_ID = ${SLURM_JOB_ID}" >> $RESDIR/results_${NODES}.txt
# echo "MKL_NUM_THREADS = $MKL_NUM_THREADS" >> $RESDIR/results_${NODES}.txt
# echo "NUMBA_THREADING_LAYER = $NUMBA_THREADING_LAYER" >> $RESDIR/results_${NODES}.txt
# echo "KMP_BLOCKTIME = $KMP_BLOCKTIME" >> $RESDIR/results_${NODES}.txt
# echo "srun -n 16 python -m tbb -p 32 --ipc ../climate.py" >> $RESDIR/results_${NODES}.txt

srun -n 16 python -m tbb -p 32 --ipc ../climate.py >> $RESDIR/results_${NODES}.txt

# for interactive -- check jupypter notebook to calc -N and -n from desired lightcone params and work size:
# salloc -N 8 -C haswell -q interactive -t 01:00:00
# srun -n 16 --tasks-per-node=2 --cpus-per-task=32 python -m tbb -p 32 --ipc ./climate.py
