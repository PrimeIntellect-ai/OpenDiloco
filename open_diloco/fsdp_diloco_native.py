import os
from contextlib import nullcontext
from typing import Literal, Optional

import torch
import wandb
from pydantic_config import parse_argv, BaseConfig
from torch.distributed import destroy_process_group, init_process_group

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from hivemind.optim.optimizer import logger


from open_diloco.utils import get_sharding_strategy

from torch.distributed.algorithms._comm_hooks.default_hooks import fp16_compress_hook, LowPrecisionState
from torch.distributed.distributed_c10d import _get_default_group


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


class Config(BaseConfig):
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    sharding_strategy: str = "NO_SHARD"
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    max_steps: int | None = None


def fp16_compress_hook_custom(state: LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor] = None):  # noqa: F821
    log("here")
    return fp16_compress_hook(state, grad, output)


def train(config: Config):
    sharding_strategy = get_sharding_strategy(config.sharding_strategy)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])

    # batch_size is the total batch size for all GPUs
    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size

    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    INPUT_DIM = 100
    OUTPUT_DIM = 100

    model = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM)
    model = model.to(local_rank)

    half_precision = config.precision == "fp16-mixed" or config.precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if config.precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16-mixed")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=half_precision_dtype) if half_precision else None,
        use_orig_params=True,
    )
    state = LowPrecisionState(process_group=_get_default_group())
    model.register_comm_hook(state=state, hook=fp16_compress_hook_custom)

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
