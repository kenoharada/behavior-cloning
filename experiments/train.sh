#!/bin/sh
. ~/venv/bc_env/bin/activate
cd ~/behavior-cloning/experiments
CUDA_VISIBLE_DEVICES=0 wandb agent {agent_id} &
PID1=$!
CUDA_VISIBLE_DEVICES=1 wandb agent {agent_id} &
PID2=$!
CUDA_VISIBLE_DEVICES=2 wandb agent {agent_id} & 
PID3=$!
CUDA_VISIBLE_DEVICES=3 wandb agent {agent_id} &
PID4=$!
CUDA_VISIBLE_DEVICES=4 wandb agent {agent_id} & 
PID5=$!
CUDA_VISIBLE_DEVICES=5 wandb agent {agent_id} &
PID6=$!
CUDA_VISIBLE_DEVICES=6 wandb agent {agent_id} &
PID7=$!
CUDA_VISIBLE_DEVICES=7 wandb agent {agent_id} &
PID8=$!

wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8