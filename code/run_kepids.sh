#!/bin/bash
# Request three nodes that will not be allocated to others for the duration of the run.
#SBATCH -N1 --exclusive

# Find out what nodes we were assigned.
srun hostname

export PATH="$HOME/miniconda/bin:$PATH"
python run_kepids.py
