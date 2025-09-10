#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --nodes=4                       # Increase for long run, e.g. 64
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00                 # Increase for long run, MAX is 24h on booster 2h on develbooster
#SBATCH --partition=devel
#SBATCH --output=%j.out

# Without this, srun does not inherit cpus-per-task from sbatch. #128
echo "START SETTING UP SBATCH"
echo "----------------------------------"

# activating the virtual environment. TODO! needs your local base path
BASE_PATH="/p/project1/nxtaim-1/neuhoefer1/CleanStableDiffusion"
source $BASE_PATH/JSC_training/activate.sh

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
echo "Job id: $SLURM_JOB_ID"

echo ""
echo "START EXECUTING MAIN"
srun python /p/data1/nxtaim/proprietary/continental/sys100_alternatives/batch_reencode.py \
            --shard_file ./partitions \
            --delete
