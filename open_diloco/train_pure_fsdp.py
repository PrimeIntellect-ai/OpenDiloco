import os
from contextlib import nullcontext
import datetime

import torch
import torch.distributed as dist
from pydantic_config import parse_argv, BaseConfig
from torch.distributed import destroy_process_group, init_process_group

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.device_mesh import init_device_mesh
from hivemind.optim.optimizer import logger

from open_diloco.utils import (
    FakeTokenizedDataset,
    get_sharding_strategy,
)

TIMEOUT_NCCL_MINUTES = os.environ.get("TIMEOUT_NCCL_MINUTES", 120)
TEST_VOCAB_SIZE = 1024


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    local_steps: int = 10


class Config(BaseConfig):
    diloco: DilocoConfig = DilocoConfig()
    path_model: str = "PrimeIntellect/llama-150m-fresh"
    torch_compile: bool = True
    attn_implementation: str = "flash_attention_2"
    # Data
    seq_length: int = 1024
    num_workers: int = 4
    # Optimization
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    warmup_steps: int = 1000
    total_steps: int = 88_000
    sharding_strategy: str = "SHARD_GRAD_OP"


def get_dataloader(tokenizer, world_size, rank, local_rank, config: Config) -> StatefulDataLoader:
    train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return StatefulDataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
    )


def get_model(config: Config) -> LlamaForCausalLM:
    # Load model
    config_model = LlamaConfig.from_pretrained(config.path_model, attn_implementation=config.attn_implementation)
    return LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=config.path_model, config=config_model)


def train(config: Config):
    sharding_strategy = get_sharding_strategy(config.sharding_strategy)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # batch_size is the total batch size for all GPUs
    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size

    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config)

    model = get_model(config)
    model = model.to(local_rank)

    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    nnodes = world_size // local_world_size
    device_mesh = init_device_mesh("cuda", (nnodes, local_world_size), mesh_dim_names=("global", "local"))

    global_pg = device_mesh.get_group("global")
    local_pg = device_mesh.get_group("local")
    log(f"global pg world : {global_pg.size()}, local pg: {local_pg.size()}")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        use_orig_params=config.torch_compile,
        process_group=local_pg,
    )
    if config.torch_compile:
        model = torch.compile(model)

    # Setup optimizers
    inner_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1, betas=(0.9, 0.95))
    # outer_optimizer = torch.optim.SGD(model.parameters(), lr=config.diloco.outer_lr, momentum=0.9, nesterov=True)

    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
    )

    model.train()

    loss_batch = 0

    train_dataloader_iterator = iter(train_dataloader)

    outer_step = 0
    while True:
        if rank == 0:
            log(f"outer_step step: {outer_step}")
        for inner_step in range(config.diloco.local_steps):
            for grad_acc_step in range(gradient_accumulation_steps):
                is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                batch = next(train_dataloader_iterator)

                for key in batch.keys():
                    batch[key] = batch[key].to("cuda")

                with model.no_sync() if is_accumulating else nullcontext():
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                    loss_batch += loss.detach()

            model.clip_grad_norm_(1.0)  # gradient clipping
            inner_optimizer.step()
            scheduler.step()
            inner_optimizer.zero_grad()

            if rank == 0:
                log(
                    f"step: {outer_step} inner: {inner_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in inner_optimizer.param_groups][0]}"
                )

            loss_batch = 0

        for param in model.parameters():  # todo make this like hybrid shard is doing
            dist.all_reduce(param, op=dist.ReduceOp.AVG, group=global_pg)

        outer_step += 1


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "PRIME_INTELLECT_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    config = Config(**parse_argv())
    train(config)
    destroy_process_group()
