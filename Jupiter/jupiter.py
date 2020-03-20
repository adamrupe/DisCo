'''
author: Adam Rupe
modified: Nalini Kumar
https://github.com/adamrupe/DisCo
brief: DisCo code for 2D turbulence data
dependencies: see environment.yml

********************************************************************************
Note that Jupiter data are unsigned ints, so can't do temporal decay; must
comment out those lines in the source code
********************************************************************************

Use the following (elsewhere, e.g. a jupyter notebook) to calculate the number
of parallel procedures needed for given choice of lightcone depths and work size
(this is the value of srun -n in run_haswell.sl):

from math import ceil
total = 220 # number of time steps in the jupiter data
p = 3 # past lightcone depth
f = 3 # future lightcone depth
work = 1 # worksize
n_procs = ceil((total - f - p)/work) # number of required parallel procedures
'''
import time, os, sys
from math import ceil

# Parent directory which contains the disco repo
module_path = os.path.abspath(os.path.join('/global/common/software/ProjectDisCo/'))
sys.path.append(module_path)

from source.pdisco import *

#Initialize MPI variables
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start = time.time()

# Initialize parameters for pipeline
depth = 3
c = 3
K_past = 8
K_future = 10
past_decay = 0
future_decay = 0
past_params = {'nClusters':K_past, 'maxIterations':200}
future_params = {'nClusters':K_future, 'maxIterations':200}

# Load data with appropriate halos
data_dir = '/global/project/projectdirs/ProjectDisCo/Jupiter/gs_arrays/'
allfiles = sorted(os.listdir(data_dir))
total = len(allfiles)
halo = depth
work = 3
n_procs = ceil((total - 2*halo)/work)
last = n_procs - 1
if rank == last:
    myfiles = allfiles[rank*work : total]
else:
    myfiles = allfiles[rank*work : (rank+1)*work + 2*halo]
myfield = np.stack([np.load(os.path.join(data_dir, f)) for f in myfiles])


# reconstruction and causal filter
recon = DiscoReconstructor(depth, depth, c)
recon.extract(myfield)
comm.Barrier()
d4p.daalinit()
recon.kmeans_lightcones(past_params, future_params, past_decay=past_decay, future_decay=future_decay)
recon.reconstruct_morphs()
comm.Barrier()
comm.Allreduce(recon.local_joint_dist, recon.global_joint_dist, op=MPI.SUM )
recon.reconstruct_states(chi_squared)
recon.causal_filter()

save_dir = '/global/cscratch1/sd/atrupe/results/Jupiter/SC19/result-3/'
first_work = myfiles[halo]
last_work = myfiles[-(halo+1)]
first_number = first_work[7:10]
last_number = last_work[7:10]
save_name = 'jup_csfield_'+first_number+'-'+last_number
np.save(save_dir+save_name, recon.state_field)

if rank == 0:
    end = time.time()
    print('Minutes to Solution: {}'.format((end-start)/60))
    print('past depth: {}, \nfuture depth: {}, \nc: {}, \npast K: {}, \nfuture K: {}, \npast decay: {}, \nfuture decay: {}'.format(depth, depth, c, K_past, K_future, past_decay, future_decay))

d4p.daalfini()
