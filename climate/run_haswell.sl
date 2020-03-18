#!/bin/bash -l

#SBATCH -p debug
##SBATCH --reservation=DisCoHackathon2
#SBATCH -t 0:30:00
#SBATCH --nodes=8
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH -C haswell
#SBATCH -J disco_ws
#SBATCH -o /global/project/projectdirs/ProjectDisCo/scaling_results/kmeans/haswell-weak/slurm_logs/%j_hsl.log
#SBATCH -e /global/project/projectdirs/ProjectDisCo/scaling_results/kmeans/haswell-weak/slurm_logs/%j_hsl.err
##SBATCH --mail-user=atrupe@ucdavis.edu
##SBATCH --mail-type=END

module load python/3.6-anaconda-4.4
module unload darshan

export MKL_NUM_THREADS=16
export NUMBA_THREADING_LAYER=TBB
export KMP_BLOCKTIME=10

source activate craympi-test

#parallel
# n = total no. of processes
# tasks-per-node = mpi processes per node

RESDIR="/global/project/projectdirs/ProjectDisCo/scaling_results/kmeans/haswell-weak/results"

NODES=8

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}" >> $RESDIR/results_${NODES}.txt
echo "MKL_NUM_THREADS = $MKL_NUM_THREADS" >> $RESDIR/results_${NODES}.txt
echo "NUMBA_THREADING_LAYER = $NUMBA_THREADING_LAYER" >> $RESDIR/results_${NODES}.txt
echo "KMP_BLOCKTIME = $KMP_BLOCKTIME" >> $RESDIR/results_${NODES}.txt
echo "srun -n 16 python -m tbb -p 32 --ipc ../climate.py" >> $RESDIR/results_${NODES}.txt

srun -n 16 python -m tbb -p 32 --ipc ../climate.py >> $RESDIR/results_${NODES}.txt

# srun -n 24 --tasks-per-node=2 --cpus-per-task=32 python -m tbb -p 32 --ipc ./climate.py