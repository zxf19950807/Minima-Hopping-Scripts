#!/bin/bash -l
#SBATCH --job-name=mh
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu
## SBATCH --gpu-bind=closest
#SBATCH --output=slurm.out
#SBATCH --time=24:00:00
#SBATCH --partition=normal
## SBATCH --time=00:30:00
## SBATCH --partition=debug
#SBATCH --account=lp08
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules

ml cray-mpich/8.1.30
ml python/3.12.5
ml cuda/12.6.2

source /users/zxingfan/bin/ase_mh/mh/bin/activate
ulimit -s unlimited

export MPICH_GPU_SUPPORT_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#export highest_utilization=0
#monitor_gpu() {
#    while true
#    do
#    utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{ sum += $1 } END { if (NR > 0) print sum / NR }')
#    if awk 'BEGIN {exit !('"$utilization"' > '"$highest_utilization"')}'; then
#            export highest_utilization=$utilization
#            echo "Current highest GPU utilization observed: ${highest_utilization}%" >> gpu.log
#        fi
#    done
#}

log_file="gpu.out"

if [ ! -f "$log_file" ]; then
    echo "Timestamp,GPU0 (%),GPU1 (%),GPU2 (%),GPU3 (%)" > "$log_file"
fi

monitor_gpu() {
    while true
    do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd "," -)
        echo "$timestamp,$utilization" >> "$log_file"
        sleep 20
    done
}

monitor_gpu &
monitor_pid=$!
export JOBREPORT=/users/zxingfan/bin/jobreport/jobreport

# Run the monitored job
srun ${JOBREPORT} -o jobreport_${SLURM_JOB_ID} --cpu-bind=socket -- ./mps-wrapper.sh python3 rh.py

# After the job is finished, print the report
${JOBREPORT} print jobreport_${SLURM_JOB_ID}

# rsync
kill $monitor_pid


