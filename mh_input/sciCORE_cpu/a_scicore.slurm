#!/bin/bash                 
#SBATCH --job-name=mh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --qos=kernph-1week --partition=kernph
#SBATCH --output=slurm.out

# miniconda
module load Python/3.11.3-GCCcore-12.3.0
module load OpenMPI/4.1.5-GCC-12.3.0
source /kernph/zhang0045/bin/mh/bin/activate

# mh
export OMP_NUM_THREADS=20
#  export MKL_NUM_THREADS=20

srun python3 rh_cpu.py
