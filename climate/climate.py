'''
author: Adam Rupe
modified: Nalini Kumar
https://github.com/adamrupe/DisCo
brief: DisCo code for CAM5.1 climate data
dependencies: see environment.yml
'''
import time, os, sys
from netCDF4 import Dataset

#Parent directory which contains the disco repo
module_path = os.path.abspath(os.path.join('/global/common/software/ProjectDisCo/'))
sys.path.append(module_path)

from source.pdisco import *

# Initialize MPI variables
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start = time.time()
    full_start = time.time()

# Initialize parameters for pipeline
past_depth = 3
future_depth = 3
c = 1
past_K = 16
future_K = 10
past_params = {'nClusters':past_K, 'maxIterations':200}
future_params = {'nClusters':future_K, 'maxIterations':200}

# Load dat with appropriate halos

run_dir = "/global/cscratch1/sd/karthik_/GB19/CAM5_TMQ_All-Hist_data/run1/"
allfiles = sorted(os.listdir(run_dir))
filenames = allfiles[7144:]
halo = 1
worksize = 6
myfiles = filenames[rank*worksize : (rank+1)*worksize + 2*halo]
myfield = np.vstack([Dataset(run_dir+ '/'+f, 'r')['TMQ'][:] for f in myfiles])
pmargin = 8 - past_depth
fmargin = 8 - future_depth
myfield = myfield[pmargin: -fmargin]

if rank == 0:
    end = time.time()
    print('load (s): {}'.format(end-start), flush=True)
    start = time.time()

recon = DiscoReconstructor(past_depth, future_depth, c)

recon.extract(myfield)
del myfield

comm.Barrier()
if rank == 0:
    end = time.time()
    print('extract (s): {}'.format((end-start)), flush=True)
    start = time.time()

d4p.daalinit()
recon.kmeans_lightcones(past_params, future_params, past_decay=0.04, future_decay=0.04)

if rank == 0:
    end = time.time()
    print('cluster (s): {}'.format((end-start)), flush=True)
    start = time.time()

recon.reconstruct_morphs()

if rank == 0:
    end = time.time()
    print('reconstruct_morphs (s): {}'.format(end-start), flush=True)

comm.Barrier()

if rank == 0:
    start = time.time()

comm.Allreduce(recon.local_joint_dist, recon.global_joint_dist, op=MPI.SUM )

if rank == 0:
    end = time.time()
    print('Allreduce (s): {}'.format(end-start), flush=True)
    start = time.time()

recon.reconstruct_states(chi_squared)

if rank == 0:
    end = time.time()
    print('reconstruct_states (s): {}'.format(end-start), flush=True)
    start = time.time()

recon.causal_filter()

if rank == 0:
    end = time.time()
    print('causal_filter (s): {}'.format(end-start), flush=True)
    start = time.time()

save_dir = '/global/cscratch1/sd/atrupe/results/climate/SC19/knl/result-1/fields/'
first_work = myfiles[halo]
np.save(save_dir+first_work[:-3]+'-{}'.format(worksize), recon.state_field)

if rank == 0:
    end = time.time()
    print('save (s): {}'.format((end-start)))

run_details = "past_depth: {} \nfuture_depth: {} \nc :{} \
               \npast_K: {} \nfuture_K: {} \nworksize: {} \nlc decay: 0.04 ".format(past_depth,
                                                                                   future_depth,
                                                                                   c,
                                                                                   past_K,
                                                                                   future_K,
                                                                                   worksize)
if rank == 0:
    full_end = time.time()
    print(run_details, flush=True)
    print('Time to solution: {}'.format(full_end-full_start))

d4p.daalfini()
