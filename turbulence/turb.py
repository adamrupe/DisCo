'''
author: Adam Rupe
modified: Nalini Kumar
https://github.com/adamrupe/DisCo
brief: DisCo code for 2D turbulence data
dependencies: see environment.yml

*******************************************************************************
Use the following (elsewhere, e.g. a jupyter notebook) to calculate the number
of parallel procedures needed for given choice of lightcone depths and work size
(this is the value of srun -n in run_haswell.sl):

from math import ceil
total = 251 # number of time steps in the turbulence data
p = 6 # past lightcone depth
f = 2 # future lightcone depth
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
p_depth = 14
f_depth = 2
c = 1
K_past = 4
K_future = 20
p_decay = 0.0
f_decay = 0.0
past_params = {'nClusters':K_past, 'maxIterations':200}
future_params = {'nClusters':K_future, 'maxIterations':200}
p_i_params = {'nClusters':K_past, 'method':'plusPlusDense', 'distributed': True}
f_i_params = {'nClusters':K_future, 'method':'plusPlusDense', 'distributed': True}

# Load data with appropriate halos
work = 2

load_file = '/global/project/projectdirs/ProjectDisCo/Turbulence/turbulence_vorticity.npy'
fullfield = np.load(load_file)

#fullfield = fullfield*fullfield #reconstructing from squared vorticity
# fullfield = np.absolute(fullfield) # reconstruct from absolute value of vorticity

# make sure the full dataset gets processed / filtered
total = len(fullfield)
n_procs = ceil((total - (p_depth+f_depth))/work)
last = n_procs - 1
if rank == last:
    myfield = np.copy(fullfield[rank*work : total])
else:
    myfield = np.copy(fullfield[rank*work : (rank+1)*work + (p_depth+f_depth)])
del fullfield

# execute distributed reconstruction and causal filtering
recon = DiscoReconstructor(p_depth, f_depth, c)
recon.extract(myfield, boundary_condition='periodic')
del myfield
comm.Barrier()
d4p.daalinit()
recon.kmeans_lightcones(past_params,
                        future_params,
                        past_decay=p_decay,
                        future_decay=f_decay,
                        past_init_params=p_i_params,
                        future_init_params=f_i_params,
                        )
recon.reconstruct_morphs()
comm.Barrier()
comm.Allreduce(recon.local_joint_dist, recon.global_joint_dist, op=MPI.SUM )
recon.reconstruct_states(chi_squared)
recon.causal_filter()

save_dir = '/global/cscratch1/sd/atrupe/results/turbulence/SC19/result-18/fields/'
save_name = 'turb_csfield_{:03d}'.format(rank)
np.save(save_dir+save_name, recon.state_field)

if rank == 0:
    end = time.time()
    print('Minutes to Solution: {}'.format((end-start)/60))
    print('past depth: {}, \nfuture depth: {}, \nc: {}, \npast K: {}, \nfuture K: {}, \npast decay: {}, \nfuture decay: {}'.format(p_depth, f_depth, c, K_past, K_future, p_decay, f_decay))
    print('Reconstructed from regular vorticity field')
    # print('Reconstructed from squared vorticity field')
    # print('Reconstructed from absolute value vorticity field')
    print('\n')

d4p.daalfini()
