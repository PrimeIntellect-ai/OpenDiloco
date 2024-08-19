# OpenDiLoCo

This repository contains the training code and experiment results for the paper [OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training](https://arxiv.org/abs/2407.07852).

# Setup

Before running the experiment scripts, you must first setup the environment.
You can clone the repository and setup a conda environment or use our pre-built docker image.

## Cloning the repository

Clone the repository along with the submodules:
```
git clone https://github.com/PrimeIntellect-ai/OpenDiLoCo.git --recursive
cd OpenDiLoCo
``` 

## Environment setup

Create a new conda environment and activate it:
```bash
conda create -n OpenDiLoCo python=3.11 -y && conda activate OpenDiLoCo
```

or with virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install python dependencies:
```bash
pip install .
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
```

Optionally, you can install flash-attn to use Flash Attention 2.
This requires your system to have cuda compiler set up.

```bash
# (Optional) flash-attn
pip install flash-attn>=2.5.8
```

## Docker container

If you prefer to run your experiments in a reproduceable container, you can use our pre-built docker image containing the repository and pre-installed dependencies.
```bash
docker pull primeintellect/open_diloco:main
docker run -d --name open-diloco --ipc=host --network=host --gpus=all primeintellect/open_diloco:main
docker exec -it open-diloco bash
```

# Experiments

This section describes the configurations we used for the experiments reported in the paper.

The scripts to launch the experiment are in the `open_diloco` folder.
The commands in this document assume you are in the `open_diloco` folder:
```bash
cd open_diloco
```

## Machine specific configurations

The torchrun arguments can be changed to match your machine configuration without affecting the final results unless stated otherwise.
The `per-device-train-batch-size` can also be changed to match the VRAM of your GPUs without affecting the final results.

## Hivemind Initialisation

Some of our experiments utilize the hivemind library to perform distributed weight averaging.
This requires a [Distributed Hash Table (DHT)](https://en.wikipedia.org/wiki/Distributed_hash_table).
An initial peer is required to initialize the DHT.
This initial peer must be reachable from all machines participating in the distributed training.

On the machine chosen as the initial peer, run:

```bash
hivemind-dht     --identity_path fixed_private_key.pem     --host_maddrs /ip4/0.0.0.0/tcp/30001
```

You should receive an output similar to this:

```bash
Feb 30 13:35:32.717 [INFO] Running a DHT instance. To connect other peers to this one, use --initial_peers /ip4/127.0.0.1/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ
Feb 30 13:35:32.717 [INFO] Full list of visible multiaddresses: /ip4/127.0.0.1/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ /ip4/192.168.100.20/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ
Feb 30 13:35:32.719 [INFO] 1 DHT nodes (including this one) are in the local routing table 
Feb 30 13:35:32.719 [INFO] Local storage contains 0 keys
```

The [multiaddress](https://github.com/multiformats/multiaddr) strings listed after `Full list of visible multiaddresses: ` in the output are the multiaddresses you can use to initialize your training processes. In this example they are `/ip4/127.0.0.1/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ` and `/ip4/192.168.100.20/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ`

## Stopping hivemind runs

The current implementation of hivemind doesn't handle Ctrl+C keyboard interrupt well. You can stop the runs using `pkill`:
```bash
pkill -f torchrun
```

## Resuming from checkpoint
To resume from checkpoint, you can pass the `--resume-from-checkpoint` argument to the training script. e.g.
```bash
torchrun --nproc_per_node=8 \
    train_fsdp.py \
    ...
    --resume-from-checkpoint checkpoints_1b/2024-06-20/hivemind_1b/bm5zjkzr/model_step_6000
```

## 150m DDP Baseline
In the `open_diloco` folder, run:
```bash
torchrun --nproc_per_node=8 \
    train_fsdp.py \
    --sharding-strategy NO_SHARD \
    --per-device-train-batch-size 32 \
    --precision bf16-mixed \
    --total-batch-size 512 \
    --total-steps 88_000 \
    --project OpenDiLoCo \
    --lr 4e-4 \
    --path_model PrimeIntellect/llama-150m-fresh \
    --log-activations-steps 200 \
    --ckpt.interval 8000 \
    --ckpt.path 150_ckpt
```

## 150m on 8 DiLoCo Worker with 500 local steps
In the `open_diloco` folder, run:
```bash
./run_training.sh 8 1 $PEER \
    --sharding-strategy NO_SHARD \
    --per-device-train-batch-size 8 \
    --precision bf16-mixed \
    --total-batch-size 512 \
    --hv.local-steps 500 \
    --total-steps 88_000 \
    --project OpenDiLoCo \
    --hv.skip_load_from_peers \
    --lr 4e-4 \
    --path-model PrimeIntellect/llama-150m-fresh \
    --log-activations-steps 250 \
    --ckpt.interval 4975  \
    --ckpt.path 150_ckpt
```

under the hood the `run_training.sh` script calls `train_fsdp.py` 8 times with the right argument to simulate 8 workers locally.


## 150m on 8 DiLoCo Worker with 50 local steps
In the `open_diloco` folder, run:
```bash
./run_training.sh 8 1 $PEER \
    --sharding-strategy NO_SHARD \
    --per-device-train-batch-size 8 \
    --total-batch-size 512 \
    --precision bf16-mixed \
    --hv.local-steps 50 \
    --total-steps 88_000 \
    --project OpenDiLoCo \
    --hv.skip_load_from_peers \
    --lr 4e-4 \
    --path-model PrimeIntellect/llama-150m-fresh \
    --log-activations-steps 250 \
    --ckpt.interval 4975  \
    --ckpt.path 150_ckpt
```

## 1b Baseline
In the `open_diloco` folder, run:
```bash
torchrun --nproc_per_node=8 \
    train_fsdp.py \
    --sharding-strategy _HYBRID_SHARD_ZERO2 \
    --per-device-train-batch-size 16 \
    --total-batch-size 8192 \
    --precision bf16-mixed \
    --total-steps 88_000 \
    --project OpenDiLoCo \
    --lr 4e-4 \
    --path_model PrimeIntellect/llama-1b-fresh \
    --ckpt.path 1b_ckpt \
    --ckpt.interval 500
```

## 1b on 4 DiLoCo Workers with 500 local steps
Set the `PEER` environment variable to the multiaddress string obtained from the **Hivemind Initialisation** step above.
Launch the command below on 4 separate machines with the environment variable `WORLD_RANK` set to `0`, `1`, `2` and `3` respectively.

```bash
export PEER=/ip4/192.168.100.20/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ
export WORLD_RANK=0

torchrun --nproc_per_node=8 \
    train_fsdp.py \
    --per-device-train-batch-size 16 \
    --total-batch-size 2048 \
    --precision bf16-mixed \
    --total-steps 88_000 \
    --hv.local_steps 500  \
    --project OpenDiLoCo \
    --lr 4e-4 \
    --path_model PrimeIntellect/llama-1b-fresh \
    --warmup-steps 1000 \
    --hv.averaging_timeout 1800 \
    --hv.skip_load_from_peers \
    --hv.local_steps 500 \
    --hv.initial-peers $PEER \
    --hv.galaxy-size 4 \
    --hv.world-rank $WORLD_RANK \
    --checkpoint_interval 500 \
    --ckpt.path 1b_diloco_ckpt
```
## 1b on 4 DiLoCo Workers with 125 local steps

similar as above but with 



```bash
export PEER=/ip4/192.168.100.20/tcp/30001/p2p/Qmbh7opLJxFCtY22XqwETuo6bnWqijs76YXz7D69MBWEuZ
export WORLD_RANK=0

torchrun --nproc_per_node=8 \
    train_fsdp.py \
    --per-device-train-batch-size 16 \
    --total-batch-size 2048 \
    --precision bf16-mixed \
    --total-steps 88_000 \
    --hv.local_steps 500  \
    --project OpenDiLoCo \
    --lr 4e-4 \
    --path_model PrimeIntellect/llama-1b-fresh \
    --warmup-steps 1000 \
    --hv.averaging_timeout 1800 \
    --hv.skip_load_from_peers \
    --hv.local_steps 125 \
    --hv.initial-peers $PEER \
    --hv.galaxy-size 4 \
    --hv.world-rank $WORLD_RANK \
    --checkpoint_interval 500 \
    --ckpt.path 1b_diloco_ckpt
```

# Use OpenDiLoCo in your own code

This codebase is composed of a full training script to use OpenDiLoCo with torch FSDP and hivemind to pretrain transformers (what is used below) as well as individual components to use OpenDiLoCo with other frameworks.

Specifically, if you want to use OpenDiLoCo in your own training script, you can replace your optimizer with `open_diloco.hivemind_diloco.DiLoCoOptimizer`, which is an (almost) drop-in replacement for `hivemind.optim.optimizer`

## Example usage of `DiLoCoOptimizer`.

```python
from functools import partial

from open_diloco.hivemind_diloco import DiLoCoOptimizer
from hivemind.dht.dht import DHT


dht = DHT(start=True, initial_peers=os.environ["PEERS"])

inner_optimizer = partial(torch.optim.AdamW, lr=4e-4)  # optimizer need to be function
outer_optimizer = partial(torch.optim.SGD, lr=0.7, momentum=0.9, nesterov=True) # optimizer need to be function

model = ...

optimizer = DiLoCoOptimizer(dht=dht,params=model.parameters(), batch_size=512, num_inner_steps=500,inner_optimizer=inner_optimizer,   outer_optimizer=outer_optimizer)

train_dataloader = ...

for step, batch in enumerate(train_dataloader):
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()

```

Note on using gradient scaler: If you are using a gradient scaler, you need to specifically call the `unscale_` on the inner optimizer.

```python
scaler.unscale_(optimizer.inner_optimizer)
```

and you need to pass the scaler as a parameter of the `optimizer.step`.

```python
optimizer.step(scaler)
```


We recommend using `bf16` to avoid scaling and desynchronization issues with hivemind/fsdp and are actively working to make it easier to handle scalers with our optimizer.


# Debugging Issues
1. `RuntimeError: CUDA error: invalid device ordinal`
    A possible culprit is that your `--nproc-per-node` argument for the torchrun launcher is set incorrectly.
    Please set it to an integer less than equal to the number of gpus you have on your machine.

2. `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate...`
    A possible culprit is that your `--per-device-train-batch-size` is too high.
    Try a smaller value.

# Citation
If you use OpenDiloco for your research, please cite our [paper](https://arxiv.org/abs/2407.07852):
```bibtex
@misc{jaghouar2024opendiloco,
    title={OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training}, 
    author={Sami Jaghouar and Jack Min Ong and Johannes Hagemann},
    year={2024},
    eprint={2407.07852},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2407.07852}, 
}
```
