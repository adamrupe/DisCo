# DisCo

Source code and run scripts for science results presented in
[DisCo: Physics-Based Unsupervised Discovery of Coherent Structures in Spatiotemporal Systems](https://arxiv.org/abs/1909.11822).

The data sets themselves are not included here, nor the code that created the visualizations.
Here we provide the source code and run scripts so that others may perform similar
local causal state analysis on their own data.

An environment.yml is provided to create the [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) we used, which contains all the necessary
Python dependencies to run the code.

Note that we ran on the Cori supercomputer, which uses Cray MPI, and therefore we
had to build daal4py from source to use Cray MPI libraries (Intel MPI is the default).

Cori uses the SLURM workload manager, so we use SLRUM scripts to run our code.

Source code for our distributed DBSCAN algorithm is not yet publicly available,
but will be added to Intel DAAL and daal4py in the future. Therefore dbscan_pdisco.py
may not run. 
