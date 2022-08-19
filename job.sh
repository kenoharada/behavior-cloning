#!/bin/sh
#$-l rt_AF=1
#$-l h_rt=72:00:00
#$-j y
#$-cwd
source /etc/profile.d/modules.sh
module load singularitypro
cd ~
# wandb agent
pids=()
singularity exec --nv ~/behavior-cloning/bc_env.sif sh ~/behavior-cloning/experiments/train.sh &
pids[$!]=$!
wait ${pids[@]}