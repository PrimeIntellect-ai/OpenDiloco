import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Literal

import fsspec
import torch
import torch.distributed as dist
import wandb
from cyclopts import App
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from fsspec.generic import GenericFileSystem
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)

from open_diloco.utils import get_grad_norm, register_hooks_log_activations


# Function to initialize the distributed process group
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# Function to load the checkpoint state into model, optimizer, and scheduler
def load_checkpoint(
    model,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    filename,
    resume_only_model: bool,
):
    with fsspec.open(filename, "rb") as f:
        checkpoint = torch.load(f)

    if resume_only_model:
        for key in list(checkpoint["model_state_dict"].keys()):
            if "module" in key:
                checkpoint["model_state_dict"][key.replace("module.", "")] = checkpoint["model_state_dict"].pop(key)

    model.load_state_dict(checkpoint["model_state_dict"])
    if not resume_only_model:
        inner_optimizer.load_state_dict(checkpoint["inner_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])

    return checkpoint["step"], checkpoint["loss"]


def save_checkpoint(
    real_step: int,
    model,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    loss,
    checkpoint_path,
    training_date,
    project,
):
    local_file_path = os.path.join(
        get_ckpt_folder(checkpoint_path, training_date, project),
        f"model_step_{real_step}.pt",
    )
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "inner_optimizer_state_dict": inner_optimizer.state_dict(),
        "outer_optimizer_state_dict": outer_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss.item(),
        "step": real_step,
    }
    with fsspec.open(local_file_path, "wb") as f:
        torch.save(checkpoint_data, f)
    print(f"Checkpoint saved at step {real_step}")


def evaluate_model(eval_dataloader, model, half_precision):
    loss_eval = 0
    step_eval = 0

    eval_start_time = time.time()
    for batch_eval in eval_dataloader:
        for key in batch_eval.keys():
            batch_eval[key] = batch_eval[key].to("cuda")

        with torch.no_grad():
            model.eval()

            with torch.autocast(device_type="cuda", dtype=torch.float16) if half_precision else nullcontext():
                outputs = model(**batch_eval)
                loss_eval += outputs.loss

        step_eval += 1

    eval_end_time = time.time()
    model.train()

    print(f"Evaluation time: {eval_end_time - eval_start_time:.2f} seconds")
    loss_eval /= step_eval
    return {"eval_loss": loss_eval, "eval_perplexity": torch.exp(loss_eval)}


def get_ckpt_folder(checkpoint_path, training_date, project):
    return os.path.join(checkpoint_path, training_date, project, wandb.run.id)


def check_checkpoint_path_access(checkpoint_path: str, training_date, project):
    dummy_file_path = os.path.join(
        get_ckpt_folder(
            checkpoint_path=checkpoint_path,
            training_date=training_date,
            project=project,
        ),
        "dummy_file.txt",
    )
    with fsspec.open(dummy_file_path, "w") as f:
        f.write("This is a dummy file for testing access.")
    gfs = GenericFileSystem()
    gfs.rm(dummy_file_path)


def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
    return [
        param.data.detach().clone().to("cpu") for group in outer_optimizer.param_groups for param in group["params"]
    ]


app = App()


@app.default
def main(
    batch_size: int = 512,
    per_device_train_batch_size: int = 32,
    seq_length: int = 1024,
    c4_tiny: bool = False,
    checkpoint_interval: int | None = None,
    checkpoint_path: str = "outputs",
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    precision: Literal["fp16-mixed", "bf16-mixed", "32-true"] = "fp16-mixed",
    project: str = "hivemind_debug",
    model_name_or_path: str = "PrimeIntellect/llama-150m-fresh",
    lr: float = 4e-4,
    resume_from_checkpoint: str | None = None,
    seed_data: int | None = None,
    eval_steps: int | None = None,
    log_activations_steps: int | None = None,
    local_steps: int = 500,
    wandb_group: str | None = None,
    resume_only_model: bool = False,
    outer_lr: float = 0.7,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # batch_size is the total batch size for all GPUs
    # assert batch_size % world_size == 0
    # batch_size = batch_size / world_size

    assert batch_size % per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    if local_rank == 0:
        wandb.init(project=project, group=wandb_group)

        training_date = datetime.now().strftime(
            "%Y-%m-%d"
        )  # we define the data at the beginning of training in case the training take several days

        check_checkpoint_path_access(checkpoint_path, training_date, project)
    # Load model configuration and tokenizer
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path).to(local_rank)

    # Setup optimizers
    inner_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    outer_optimizer = torch.optim.SGD(model.parameters(), lr=outer_lr, momentum=0.9, nesterov=True)

    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if precision not in ["fp16-mixed", "bf16-mixed", "32-true"]:
        raise ValueError(f"Invalid precision: {precision}. Please choose 'fp16-mixed', 'bf16-mixed', or '32-true'.")

    half_precision = precision == "fp16-mixed" or precision == "bf16-mixed"
    half_precision_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16-mixed")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it

    ds = (
        load_dataset("PrimeIntellect/c4-tiny", "en", ignore_verifications=True)
        if c4_tiny
        else load_dataset(
            "allenai/c4",
            "en",
            streaming=True,
            data_files={
                "train": "en/c4-train.*.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
        )
    )
    # we only load one eval file to be faster

    if seed_data is not None:
        ds = ds.shuffle(seed=seed_data)

    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length)
        return outputs

    tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = split_dataset_by_node(tokenized_datasets["train"], world_size=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=per_device_train_batch_size)

    if eval_steps is not None:
        eval_dataset = tokenized_datasets["validation"]
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
        )

    start_step = 0

    if resume_from_checkpoint is not None:
        last_step, last_loss = load_checkpoint(
            model,
            inner_optimizer,
            outer_optimizer,
            scheduler,
            resume_from_checkpoint,
            resume_only_model,
        )
        start_step = last_step
        print(f"Resumed from checkpoint at step {start_step} with loss {last_loss}")

    for param in model.parameters():
        # this make sure all device have the same weight init
        dist.broadcast(param.data, src=0)

    params_offloaded = get_offloaded_param(outer_optimizer)

    model.train()

    log_activations = None

    start_time = time.time()
    print(f"starting from step {start_step}")

    check_start_step = start_step > 0

    handles = None

    loss_batch = 0

    for step, batch in enumerate(iterable=train_dataloader):
        real_step = (step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (step + 1) % gradient_accumulation_steps

        if check_start_step:
            if real_step < start_step:
                continue  # skipping steps before start_step
            else:
                check_start_step = False
                print(f"skipped step {step+1}, real_step {real_step} in {time.time() - start_time:.2f} seconds")
                continue

        if log_activations_steps is not None:
            if (
                real_step >= log_activations_steps
                and real_step % log_activations_steps == 0
                and step_within_grad_acc == 0
            ):
                if local_rank == 0:
                    print(f"Logging activations at step {real_step}")
                handles, log_activations = register_hooks_log_activations(model)

            if (
                real_step - 1 >= log_activations_steps
                and (real_step - 1) % log_activations_steps == 0
                and step_within_grad_acc == 0
            ):
                if local_rank == 0:
                    print(f"Removing activations logging at step {real_step}")

                # if we are after the step where we log the activations, we remove the hooks
                if handles is not None:
                    for handle in handles:
                        handle.remove()
                handles = None
                log_activations = None

        for key in batch.keys():
            batch[key] = batch[key].to("cuda")

        with torch.autocast(device_type="cuda", dtype=half_precision_dtype) if half_precision else nullcontext():
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps

            loss_batch += loss.detach()

        scaler.scale(loss).backward()

        if step_within_grad_acc == 0:
            scaler.unscale_(optimizer=inner_optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

            scaler.step(optimizer=inner_optimizer)
            scaler.update()
            scheduler.step()

            if log_activations_steps is not None and real_step % log_activations_steps == 0:
                log_norms_data = get_grad_norm(model)
            else:
                log_norms_data = None

            inner_optimizer.zero_grad()

            if real_step % local_steps == 0:
                if local_rank == 0:
                    print(f"perform outer step at step {real_step}")

                main_param = [param for group in inner_optimizer.param_groups for param in group["params"]]

                for param_offloaded, param in zip(params_offloaded, main_param):
                    param_offloaded_on_device = param_offloaded.data.to(param.device)
                    param.grad = param_offloaded_on_device - param.data
                    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
                    param.data = param_offloaded_on_device

                # here we don't call scaler.step. Indeed the scaler has already done his work (scaling down the gradients) with the optimizer.step call
                outer_optimizer.step()

                outer_optimizer.zero_grad()

                params_offloaded = get_offloaded_param(outer_optimizer)

            if local_rank == 0 and eval_steps is not None and real_step % eval_steps == 0:
                print(f"Evaluating at step {real_step}")

                if handles is not None:
                    for handle in handles:
                        handle.remove()
                    handles = None
                dict_to_log_eval = evaluate_model(eval_dataloader, model, half_precision)

            else:
                dict_to_log_eval = {}

            if local_rank == 0:
                dict_to_log = {
                    "Loss": loss_batch.item(),
                    "step": real_step,
                    "lr": [group["lr"] for group in inner_optimizer.param_groups][0],
                    "Perplexity": torch.exp(loss_batch).item(),
                    "effective_step": real_step * world_size,
                    "total_samples": real_step * batch_size * world_size,
                }
                dict_to_log.update(dict_to_log_eval)

                if log_norms_data is not None:
                    dict_to_log.update(log_norms_data)

                if log_activations:
                    for key, _ in log_activations.items():
                        log_activations[key] /= gradient_accumulation_steps
                        # log activation will accumulate all of the norm of the activations at each grad acc step
                        # so we need to divide
                    dict_to_log.update(log_activations)

                wandb.log(dict_to_log)
                print(
                    f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in inner_optimizer.param_groups][0]}"
                )
                loss_batch = 0

            # Save checkpoint every 'checkpoint_interval' steps
            if local_rank == 0 and checkpoint_interval is not None and real_step % checkpoint_interval == 0:
                print(f"saving at step {real_step}, step {step+1}")
                save_checkpoint(
                    real_step,
                    model,
                    inner_optimizer,
                    outer_optimizer,
                    scheduler,
                    loss,
                    checkpoint_path,
                    training_date,
                    project,
                )

    print("Training completed.")
    wandb.finish()


if __name__ == "__main__":
    ddp_setup()
    app()
    destroy_process_group()
