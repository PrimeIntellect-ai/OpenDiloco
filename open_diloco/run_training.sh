#!/bin/bash

## example usage
# ./run_training.sh 2 /ip4/127.0.0.1/tcp/36593/p2p/12D3KooWEAyutJ1zFqhAbzDn1LSzraB3o1uS8GSHxQYM87QP4AHN --per-device-train-batch-size 16 --batch-size 512 --local-steps 10  --total-steps 88000 --c4-tiny
# note that everything after the initial peer with will pass to all worker

# Check if at least two arguments were passed
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <N> <initial_peer> [additional_python_args]"
    exit 1
fi

N=$1         # Set N from the first argument
INITIAL_PEER=$2  # Set INITIAL_PEER from the second argument
shift 2      # Remove the first two arguments so $@ contains only additional Python arguments

# Ensure the logs directory exists
mkdir -p logs

# Execute the command for the first device (CUDA_VISIBLE_DEVICES=0) and log the output, run in background
echo "Command: CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 train_fsdp.py  --initial-peers $INITIAL_PEER $@ "
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 train_fsdp.py --hv.initial-peers $INITIAL_PEER $@ --hv.world-rank 0 --hv.galaxy-size $N  > logs/log0 2>&1 &
# Wait for 1 second before continuing with the rest
sleep 2

# Loop from 1 to N-1 and execute the command with different CUDA_VISIBLE_DEVICES and seed values, logging each command's output, run each in background
for i in $(seq 1 $(($N - 1)))
do
  echo "Command: CUDA_VISIBLE_DEVICES=$i torchrun --nproc_per_node=1 --nnodes=1 train_fsdp.py.py --initial-peers $INITIAL_PEER $@"
  WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$i torchrun --nproc_per_node=1 --rdzv-endpoint localhost:123$i --nnodes=1 train_fsdp.py --hv.initial-peers $INITIAL_PEER $@ --hv.world-rank $i --hv.galaxy-size $N > logs/log$i 2>&1 &
done

tail -f logs/log0
