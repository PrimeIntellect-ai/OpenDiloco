import os
from contextlib import nullcontext
from typing import Literal, Optional

import torch
import wandb
from pydantic_config import parse_argv, BaseConfig
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from hivemind.optim.optimizer import logger


from open_diloco.utils import get_sharding_strategy

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.algorithms._comm_hooks.default_hooks import DefaultState, reduce_scatter_hook, allreduce_hook


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


class DilocoConfig(BaseConfig):
    num_inner_steps: int = 5


class Config(BaseConfig):
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    sharding_strategy: str = "NO_SHARD"
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    max_steps: int | None = None
    n_layers: int = 4
    diloco: DilocoConfig = DilocoConfig()


class DilocoState:
    """
    DilocoState is used to determine whether sync the (pseudo-)gradient globally or the gradient locally.

    Args:
        num_inner_steps: int - the number of inner steps
        device_mesh: DeviceMesh - the device mesh to describe the communication topology. Should have only two dimension.
            inter-node is the outermost dimension, and the intra-node is the innermost dimension.

    """

    __slots__ = ["current_step", "num_inner_steps", "intra_node_comm_state", "inter_node_comm_state"]

    def __init__(self, num_inner_steps: int, device_mesh: DeviceMesh):
        self.current_step = 0
        self.num_inner_steps = num_inner_steps

        assert isinstance(device_mesh, DeviceMesh) and device_mesh.ndim == 2

        self.intra_node_comm_state = DefaultState(process_group=device_mesh.get_group(mesh_dim=1))
        print(f"intra_node_comm_state: {dist.get_world_size(self.intra_node_comm_state.process_group)}")
        self.inter_node_comm_state = DefaultState(process_group=device_mesh.get_group(mesh_dim=0))
        print(f"inter_node_comm_state: {dist.get_world_size(self.inter_node_comm_state.process_group)}")

    @property
    def should_sync_globally(self):
        return self.current_step % self.num_inner_steps == 0

    def increment_current_step(self):
        if self.current_step == self.num_inner_steps:
            self.current_step = 0
        else:
            self.current_step += 1


def diloco_comm_hook(state: DilocoState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):
    comm_state = state.intra_node_comm_state if state.should_sync_globally else state.inter_node_comm_state

    if state.should_sync_globally:
        log(f"Syncing global state {state.current_step}")
    else:
        log(f"Syncing local state {state.current_step} / {state.num_inner_steps}")

    if output is not None:
        reduce_scatter_hook(comm_state, grad, output)
    else:
        allreduce_hook(comm_state, grad)

    state.increment_current_step()


def train(config: Config):
    sharding_strategy = get_sharding_strategy(config.sharding_strategy)
    _local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    _rank = int(os.environ["RANK"])

    # batch_size is the total batch size for all GPUs
    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size

    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    INPUT_DIM = 100
    OUTPUT_DIM = 100

    model = torch.nn.Sequential(*(torch.nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=True) for _ in range(config.n_layers)))
    model = model.to("cuda")

    half_precision = config.precision == "fp16-mixed" or config.precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if config.precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16-mixed")

    nnodes = world_size // local_world_size
    devices_mesh_diloco = DeviceMesh(
        "cuda",
        mesh=[[i * world_size + j for j in range(world_size)] for i in range(nnodes)],
    )
    print(f"devices_mesh_diloco: {devices_mesh_diloco}")

    if sharding_strategy in [
        ShardingStrategy._HYBRID_SHARD_ZERO2,
        ShardingStrategy.HYBRID_SHARD,
    ]:
        device_mesh = devices_mesh_diloco
    else:
        device_mesh = None

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=half_precision_dtype) if half_precision else None,
        use_orig_params=True,
        device_mesh=device_mesh,
    )
    state = DilocoState(num_inner_steps=config.diloco.num_inner_steps, device_mesh=devices_mesh_diloco)
    model.register_comm_hook(state=state, hook=diloco_comm_hook)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1, betas=(0.9, 0.95))

    model.train()

    loss_batch = 0

    for step in range(config.max_steps):
        real_step = (step + 1) // gradient_accumulation_steps
        is_accumulating = bool((step + 1) % gradient_accumulation_steps)

        input = torch.rand(config.per_device_train_batch_size, INPUT_DIM).to("cuda")
        target = torch.randint(0, OUTPUT_DIM, (config.per_device_train_batch_size,)).to("cuda")

        with model.no_sync() if is_accumulating else nullcontext():
            outputs = model(input)
            loss = torch.nn.functional.cross_entropy(outputs, target)
            loss = loss / gradient_accumulation_steps
            loss_batch += loss.detach()
            scaler.scale(loss).backward()

        if not is_accumulating:
            scaler.unscale_(optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            log(f"step {step}, loss {loss_batch}")
            loss_batch = 0

        if config.max_steps is not None and real_step >= config.max_steps:
            break
    log("Training completed.")
    wandb.finish()


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    config = Config(**parse_argv())
    train(config)
    destroy_process_group()
