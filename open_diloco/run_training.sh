#!/bin/bash

## example usage
# ./run_training.sh 4 2 /ip4/127.0.0.1/tcp/36593/p2p/12D3KooWEAyutJ1zFqhAbzDn1LSzraB3o1uS8GSHxQYM87QP4AHN --per-device-train-batch-size 16 --batch-size 512 --local-steps 10  --total-steps 88000 --c4-tiny
# note that everything after the initial peer with will pass to all worker
#
## the command above will use a total of 8 gpu and create 4 diloco workers each of them with two gpu training ddp/fsdp wise


# you can either pass a fixed initial peer or set it to auto and the script will start a dht server for you
## ./run_training.sh 4 1 auto  --per-device-train-batch-size 8 --total-batch-size 128 --lr 1e-2 --path-model ../tests/models/llama-2m-fresh --project debug  --no-torch-compile --hv.local-steps 100  --fake-data --hv.matchmaking_time 2 --hv.fail_rank_drop --hv.skip_load_from_peers

# Function to get CUDA devices based on the number of GPUs and index
function get_cuda_devices() {
    local num_gpu=$1
    local index=$2
    local start_gpu=$((num_gpu * index))
    local end_gpu=$((start_gpu + num_gpu - 1))

    if [ "$num_gpu" -eq 1 ]; then
        echo $start_gpu
    else
        echo $(seq -s ',' $start_gpu $end_gpu)
    fi
}

# Check if at least three arguments were passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <N> <initial_peer> <num_gpu> [additional_python_args]"
    exit 1
fi

N=$1         # Set N from the first argument
NUM_GPU=$2
INITIAL_PEER=$3  # Set INITIAL_PEER from the second argument
shift 3      # Remove the first three arguments so $@ contains only additional Python arguments

mkdir -p logs
echo "Initial peer: $INITIAL_PEER"

# Check if INITIAL_PEER is set to 'auto' and adjust accordingly
if [ "$INITIAL_PEER" = "auto" ]; then
    # start the dht server
    echo "Starting DHT server"
    hivemind-dht --host_maddr /ip4/0.0.0.0/tcp/12345 --identity_path fixed_key.pem  > logs/log_dht 2>&1 &
    
    INITIAL_PEER=""
    # get the initial peer from the logs, loop until the peer is found
    while [ -z "$INITIAL_PEER" ]; do
        sleep 1
        INITIAL_PEER=$(awk '/Running a DHT instance/ {print $NF}' logs/log_dht)

    done
fi
echo "Initial peer: $INITIAL_PEER"

# Ensure the logs directory exists
mkdir -p logs

# Execute the command for the first device and log the output, run in background
CUDA_VISIBLE_DEVICES=$(get_cuda_devices $NUM_GPU 0) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 train_fsdp.py --hv.initial-peers $INITIAL_PEER $@ --hv.world-rank 0 --hv.galaxy-size $N  > logs/log0 2>&1 &
# Wait for 1 second before continuing with the rest
sleep 2

# Loop from 1 to N-1 and execute the command with different CUDA_VISIBLE_DEVICES and seed values, logging each command's output, run each in background
for i in $(seq 1 $(($N - 1)))
do
  WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$(get_cuda_devices $NUM_GPU $i) torchrun --nproc_per_node=$NUM_GPU --rdzv-endpoint localhost:123$i --nnodes=1 train_fsdp.py --hv.initial-peers $INITIAL_PEER $@ --hv.world-rank $i --hv.galaxy-size $N > logs/log$i 2>&1 &
done

tail -f logs/log0
