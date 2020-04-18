#!/bin/bash
# Request three nodes that will not be allocated to others for the duration of the run.
#SBATCH -N1 --exclusive

# Find out what nodes we were assigned.
srun hostname

module purge
module add gcc python3
echo "Using: $(which python3)"

python3 run_kepids.py
