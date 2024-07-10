"""

to test quickly do 
torchrun --nproc_per_node=2 \
        train_fsdp.py --per-device-train-batch-size 8 --total-batch-size 128 --lr 1e-2 --path-model ../tests/models/llama-2m-fresh \
        --no-torch-compile --log-activations-steps 5 --fake-data --max-steps 20
"""

from functools import partial
import os
import time
from contextlib import nullcontext
import datetime
from typing import Any, Literal

import fsspec
from pydantic import model_validator
import torch
import wandb
from pydantic_config import parse_argv, BaseConfig
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from fsspec.generic import GenericFileSystem
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
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed import broadcast_object_list
from open_diloco.ckpt_utils import load_checkpoint, save_checkpoint
from open_diloco.hivemind_diloco import AllReduceStrategy, DiLoCoOptimizer


from hivemind.dht.dht import DHT
from hivemind.utils.networking import log_visible_maddrs
from hivemind.optim.optimizer import logger


from open_diloco.utils import (
    ActivationNormMetric,
    FakeTokenizedDataset,
    get_compression_kwargs,
    get_sharding_strategy,
)


TIMEOUT_NCCL_MINUTES = os.environ.get("TIMEOUT_NCCL_MINUTES", 120)
TARGET_LAYER_ACTIVATIONS = ["self_attn", "lm_head"]
TEST_VOCAB_SIZE = 1024


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=TIMEOUT_NCCL_MINUTES))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log(message):
    logger.info(f"[rank {os.environ['LOCAL_RANK']}] {message}")


def get_ckpt_folder(checkpoint_path, training_date, project, run_id):
    return os.path.join(checkpoint_path, training_date, project, run_id)


def check_checkpoint_path_access(checkpoint_path: str, training_date, project, run_id, rank):
    dummy_file_path = os.path.join(
        get_ckpt_folder(
            checkpoint_path=checkpoint_path,
            training_date=training_date,
            project=project,
            run_id=run_id,
        ),
        f"dummy_file_{rank}.txt",
    )
    with fsspec.open(dummy_file_path, "w") as f:
        f.write("This is a dummy file for testing access.")
    gfs = GenericFileSystem()
    gfs.rm(dummy_file_path)


class HvConfig(BaseConfig):
    outer_lr: float = 0.7
    local_steps: int = 500
    initial_peers: list[str] | None = None
    host_maddrs: list[str] = ["/ip4/0.0.0.0/tcp/0"]
    announce_maddrs: list[str] | None = None
    matchmaking_time: float | None = None
    averaging_timeout: float | None = None
    hivemind_compression: Literal["none", "fp16", "scaled-fp16"] = "none"
    all_reduce_strategy: AllReduceStrategy = AllReduceStrategy.WAIT_FOR_ALL
    timeout_waiting_for_peers: float | None = None
    skip_load_from_peers: bool = False
    world_rank: int
    galaxy_size: int

    @model_validator(mode="before")
    def cast_str_to_list(cls, values: dict[str, Any]) -> dict[str, Any]:
        """This allow to only pass a string and it will still be cast as a list"""
        for arg_name in ["initial_peers", "host_maddrs", "announce_maddrs"]:
            if arg_name in values.keys() and isinstance(values[arg_name], str):
                values[arg_name] = [values[arg_name]]
        return values


class Config(BaseConfig):
    path_model: str = "PrimeIntellect/llama-150m-fresh"
    torch_compile: bool = True
    attn_implementation: str = "sdpa"
    # Data
    dataset_name_or_path: str = "allenai/c4"
    seq_length: int = 1024
    c4_tiny: bool = False
    num_workers: int = 4
    # Optimization
    lr: float = 4e-4
    total_batch_size: int = 512
    per_device_train_batch_size: int = 32
    warmup_steps: int = 1000
    total_steps: int = 88_000
    sharding_strategy: str = "NO_SHARD"
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed"
    # Checkpointing and logging
    project: str = "hivemind_debug"
    log_activations_steps: int | None = None
    resume_from_checkpoint: str | None = None
    checkpoint_interval: int | None = None
    checkpoint_path: str = "outputs"
    # Hivemind
    hv: HvConfig | None = None  # if no hv config then hivemind is disabled
    fake_data: bool = False
    max_steps: int | None = None


def get_dataloader(tokenizer, world_size, rank, local_rank, config: Config) -> StatefulDataLoader:
    if config.fake_data:
        train_dataset = FakeTokenizedDataset(config.seq_length, TEST_VOCAB_SIZE)
    else:
        ds = load_dataset(config.dataset_name_or_path, "en", streaming=True)

        def tokenize_function(data):
            outputs = tokenizer(
                data["text"],
                truncation=True,
                max_length=config.seq_length,
                padding="max_length",
            )
            return outputs

        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])[
            "train"
        ]

        if config.hv is not None:
            train_dataset = split_dataset_by_node(
                tokenized_datasets,
                world_size=config.hv.galaxy_size * world_size,
                rank=config.hv.world_rank * world_size + local_rank,
            )

        else:
            train_dataset = split_dataset_by_node(tokenized_datasets, world_size=world_size, rank=rank)

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

    world_messenger_hv = config.hv is not None and local_rank == 0

    # batch_size is the total batch size for all GPUs
    assert config.total_batch_size % world_size == 0
    batch_size = config.total_batch_size // world_size

    assert batch_size % config.per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // config.per_device_train_batch_size

    training_date = datetime.datetime.now().strftime(
        "%Y-%m-%d"
    )  # we define the data at the beginning of training in case the training take several days

    if config.hv is not None:
        sharding_strategy = ShardingStrategy.NO_SHARD
        log("Hivemind is used, ShardingStrategy.NO_SHARD is used")

    run_id = None
    if rank == 0:
        wandb.init(project=config.project, config=config.model_dump())
        run_id = wandb.run.id

    run_id_list = [run_id]
    broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]

    if config.hv is not None:
        log("hivemind diloco enabled")

    if world_messenger_hv:
        dht = DHT(
            start=True,
            initial_peers=config.hv.initial_peers,
            host_maddrs=config.hv.host_maddrs,
            announce_maddrs=config.hv.announce_maddrs,
        )
        log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=False)

    if local_rank == 0:
        check_checkpoint_path_access(config.checkpoint_path, training_date, config.project, run_id, rank)

    # DataLoader preparation
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    train_dataloader = get_dataloader(tokenizer, world_size, rank, local_rank, config)

    model = get_model(config)
    model = model.to(local_rank)

    half_precision = config.precision == "fp16-mixed" or config.precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if config.precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16-mixed")

    if sharding_strategy in [
        ShardingStrategy._HYBRID_SHARD_ZERO2,
        ShardingStrategy.HYBRID_SHARD,
    ]:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        nnodes = world_size // local_world_size
        device_mesh = DeviceMesh(
            "cuda",
            mesh=[[i * local_world_size + j for j in range(local_world_size)] for i in range(nnodes)],
        )
    else:
        device_mesh = None
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=half_precision_dtype) if half_precision else None,
        use_orig_params=config.torch_compile,
        device_mesh=device_mesh,
    )
    if config.torch_compile:
        model = torch.compile(model)

    # Setup optimizers
    inner_optimizer = partial(torch.optim.AdamW, lr=config.lr, weight_decay=0.1, betas=(0.9, 0.95))  # noqa: F821

    if config.hv is not None:
        outer_optimizer = partial(torch.optim.SGD, lr=config.hv.outer_lr, momentum=0.9, nesterov=True)

    def scheduler_fn(opt):
        return get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

    if config.hv is not None:
        if config.resume_from_checkpoint:
            # We need to load with a fake optimizer to set the model parameters correctly before initializing the DiLoCoOptimizer
            # This is because the DiLoCoOptimizer makes a copy of the model parameters for the state averager which is hard to update later
            # We also need to do this on follower workers so that the world_messenger has friends to talk to when it does its two loads
            # Otherwise the world messenger will get lonely and hang
            fake_optimizer = inner_optimizer(model.parameters())
            last_loss = load_checkpoint(
                checkpoint_path=config.resume_from_checkpoint,
                model=model,
                optimizer=fake_optimizer,
            )
            del fake_optimizer

    if world_messenger_hv:
        diloco_args = dict(
            dht=dht,
            run_id="llama",
            batch_size=batch_size,
            num_inner_steps=config.hv.local_steps,
            outer_optimizer=outer_optimizer,
            inner_optimizer=inner_optimizer,
            scheduler=None,
            params=model.parameters(),
            delay_optimizer_step=False,
            delay_grad_averaging=False,
            verbose=True,
            all_reduce_strategy=config.hv.all_reduce_strategy,
            timeout_waiting_for_peers=config.hv.timeout_waiting_for_peers,
        )

        diloco_args.update(get_compression_kwargs(config.hv.hivemind_compression))

        if config.hv.averaging_timeout is not None:
            diloco_args["averaging_timeout"] = config.hv.averaging_timeout

        if config.hv.matchmaking_time is not None:
            diloco_args["matchmaking_time"] = config.hv.matchmaking_time

        optimizer = DiLoCoOptimizer(**diloco_args)

        scheduler = scheduler_fn(
            optimizer.inner_optimizer
        )  # scheduler(optimizer) should work but better to make it explicit here

        if config.resume_from_checkpoint:
            last_loss = load_checkpoint(
                checkpoint_path=config.resume_from_checkpoint,
                model=model,
                optimizer=optimizer.inner_optimizer,
                scheduler=scheduler,
                outer_optimizer=optimizer.state_averager.optimizer,
                scaler=scaler,
                data_loader=train_dataloader,
            )
            start_step = scheduler.last_epoch
        else:
            start_step = 0

    else:
        optimizer = inner_optimizer(model.parameters())
        scheduler = scheduler_fn(optimizer)
        if config.resume_from_checkpoint:
            last_loss = load_checkpoint(
                checkpoint_path=config.resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_loader=train_dataloader,
            )
            start_step = scheduler.last_epoch
        else:
            start_step = 0

    if config.resume_from_checkpoint:
        log(f"Resumed from checkpoint at step {start_step} with loss {last_loss}")

    model.train()

    if world_messenger_hv and not config.hv.skip_load_from_peers:
        optimizer.load_state_from_peers()

    current_time = time.time()
    log(f"starting from step {start_step}")

    loss_batch = 0

    for step, batch in enumerate(iterable=train_dataloader, start=start_step * gradient_accumulation_steps):
        real_step = (step + 1) // gradient_accumulation_steps
        is_accumulating = bool((step + 1) % gradient_accumulation_steps)

        logging_activations_steps = (
            config.log_activations_steps is not None and real_step % config.log_activations_steps == 0
        )

        if logging_activations_steps:
            activation_monitor = ActivationNormMetric(
                target_layers=TARGET_LAYER_ACTIVATIONS,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            activation_monitor.register_metrics_hooks(model)

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        with model.no_sync() if is_accumulating else nullcontext():
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps

            loss_batch += loss.detach()

            scaler.scale(loss).backward()

        if not is_accumulating:
            if world_messenger_hv:
                scaler.unscale_(optimizer=optimizer.inner_optimizer)
            else:
                scaler.unscale_(optimizer=optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

            if world_messenger_hv:
                optimizer.step(scaler=scaler)

                # todo(sami): refactor to use built in pytorch mechanism to handle scaler manually
                # should allow to just do scaler.step(optimizer)
            else:
                scaler.step(optimizer)

            scaler.update()

            scheduler.step()
            optimizer.zero_grad()

            if logging_activations_steps:
                activation_monitor.remove_hooks()

            if config.hv is not None:
                if int(real_step) % config.hv.local_steps == 0:
                    for param in model.parameters():
                        torch.distributed.broadcast(param.data, src=0)

            if rank == 0:
                total_samples = real_step * config.total_batch_size
                effective_step = real_step

                if config.hv is not None:
                    # Note that this assumes that we have the right amount of worker since t0.
                    # Not robust to off/on ramping
                    effective_step = real_step * config.hv.galaxy_size
                    total_samples = real_step * config.total_batch_size * config.hv.galaxy_size

                metrics = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": effective_step,  # at each step the we have compute total_batch_size. Independent of the number of GPUs
                    "total_samples": total_samples,
                    "time_taken": time.time() - current_time,
                    "tokens_per_second": config.seq_length * config.total_batch_size / (time.time() - current_time),
                }

                if world_messenger_hv:
                    outer_lr = [group["lr"] for group in optimizer.state_averager.optimizer.param_groups][0]
                    num_peers = optimizer.tracker.global_progress.num_peers
                    if num_peers == 0:
                        num_peers = 1

                    metrics["outer_lr"] = outer_lr
                    metrics["num_peers"] = num_peers

                if logging_activations_steps:
                    metrics.update(activation_monitor.log_activations)

                current_time = time.time()

                wandb.log(metrics)

                if config.hv is None:
                    log(
                        f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in optimizer.param_groups][0]}"
                    )

            # Save checkpoint every 'checkpoint_interval' steps
            if config.checkpoint_interval is not None and real_step % config.checkpoint_interval == 0:
                log(f"saving at step {real_step}, step {step+1}")
                ckpt_path = os.path.join(
                    get_ckpt_folder(config.checkpoint_path, training_date, config.project, run_id),
                    f"model_step_{int(real_step)}",
                )

                if world_messenger_hv:
                    assert isinstance(optimizer, DiLoCoOptimizer)
                    with optimizer.tracker.pause_updates():
                        save_checkpoint(
                            checkpoint_path=ckpt_path,
                            model=model,
                            optimizer=optimizer.inner_optimizer,
                            scheduler=scheduler,
                            outer_optimizer=optimizer.state_averager.optimizer,
                            loss=loss_batch.item(),
                            scaler=scaler,
                            data_loader=train_dataloader,
                            save_global_state=True,
                        )
                else:
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=loss_batch.item(),
                        scaler=scaler,
                        data_loader=train_dataloader,
                        save_global_state=rank == 0,
                    )

            loss_batch = 0

            if config.max_steps is not None and real_step >= config.max_steps:
                break
    log("Training completed.")
    wandb.finish()


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "PRIME_INTELLECT_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    ddp_setup()
    config = Config(**parse_argv())
    train(config)
    destroy_process_group()
