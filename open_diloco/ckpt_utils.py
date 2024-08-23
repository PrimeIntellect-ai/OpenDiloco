import fsspec
from pydantic_config import BaseConfig
import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
import os
from torchdata.stateful_dataloader import StatefulDataLoader
from fsspec.generic import GenericFileSystem
from hivemind.optim.optimizer import logger


GLOBAL_STATE_FILE = "global_state_dict.pt"
CKPT_PREFIX = "model_step"


class CkptConfig(BaseConfig):
    resume: str | bool | None = None  # if resume is a boolean, it means we should resume from the last checkpoint
    interval: int | None = None
    path: str = "outputs"
    topk: int | None = None  # how many checkpoints to keep


def get_resume_info(ckpt_config: CkptConfig) -> tuple[bool, str | None]:
    """
    check if we should resume from a checkpoint, if yes return the path to the checkpoint, otherwise return None
    """
    if ckpt_config.resume is None:
        return False, None
    elif isinstance(ckpt_config.resume, bool):
        # Using fsspec to list directory contents
        fs = GenericFileSystem()
        try:
            ckpt_files = [f for f in fs.ls(ckpt_config.path, detail=False) if filter_ckpt_files(f)]
        except FileNotFoundError:
            logger.info(f"Checkpoint path {ckpt_config.path} not found, starting from scratch")
            return False, None

        if len(ckpt_files) == 0:
            logger.info(f"No checkpoints found in {ckpt_config.path}, starting from scratch")
            return False, None

        latest_ckpt = max(ckpt_files, key=lambda f: int(f.split("_")[-1]))
        return True, latest_ckpt
    else:
        return True, ckpt_config.resume


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    outer_optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    loss: float | None = None,
    data_loader: StatefulDataLoader | None = None,
    save_global_state: bool = True,
):
    """Save the model and optimizer state to a checkpoint folderx

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to save
        optimizer: the optimizer to save
        scheduler: the scheduler to save
        outer_optimizer: the outer optimizer to save
        loss: the loss to save
        data_loader: the data loader to save
        save_global_state: whether to save the global state
    """
    rank = int(os.environ["RANK"])

    # 1. Save distributed states
    fs_storage_writer = dcp.FsspecWriter(checkpoint_path, sync_files=False)
    # for some reason sync_files = True try to call stream.fileno which is not supported with gcp ffspec storage.

    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }
    dcp.save(dcp_state_dict, storage_writer=fs_storage_writer)
    if data_loader is not None:
        rank_state_dict = {}
        rank_state_dict["data_loader"] = data_loader.state_dict()
        with fsspec.open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "wb") as f:
            torch.save(rank_state_dict, f)

    if not save_global_state:
        return

    # 2. Save global states
    global_state_dict = {"scheduler": scheduler.state_dict(), "loss": loss if loss is not None else 0}
    if outer_optimizer is not None:
        global_state_dict["outer_optimizer"] = outer_optimizer.state_dict()
    if scaler is not None:
        global_state_dict["scaler"] = scaler.state_dict()

    with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "wb") as f:
        torch.save(global_state_dict, f)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    outer_optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    data_loader: StatefulDataLoader | None = None,
) -> float:
    """Load the model and optimizer state from a checkpoint folder

    Args:
        checkpoint_path: the path to the checkpoint folder
        model: the model to load
        optimizer: the optimizer to load
        scheduler: the scheduler to load
        outer_optimizer: the outer optimizer to load
        data_loader: the data loader to load

    Returns:
        loss: the loss from the checkpoint
    """
    rank = int(os.environ["RANK"])
    # 1. Load distributed states
    fs_storage_reader = dcp.FsspecReader(checkpoint_path)

    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    dcp_state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
    }
    dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizer_state_dict,
    )
    if data_loader is not None:
        with fsspec.open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "rb") as f:
            rank_state_dict = torch.load(f)
        data_loader.load_state_dict(rank_state_dict["data_loader"])

    # 2. Load global states
    with fsspec.open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "rb") as f:
        global_state_dict = torch.load(f)
    if scheduler is not None:
        scheduler.load_state_dict(global_state_dict["scheduler"])
        optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
    if outer_optimizer is not None:
        outer_optimizer.load_state_dict(global_state_dict["outer_optimizer"])
    if scaler is not None:
        scaler.load_state_dict(global_state_dict["scaler"])
    return global_state_dict["loss"]


def filter_ckpt_files(f):
    if CKPT_PREFIX not in f:
        return False
    else:
        try:
            int(f.split("_")[-1])
            return True
        except ValueError:
            return False


def delete_old_checkpoints(checkpoint_path: str, topk: int) -> list[str]:
    fs = GenericFileSystem()
    ckpt_files = [f for f in fs.ls(checkpoint_path, detail=False) if filter_ckpt_files(f)]
    ckpt_files.sort(key=lambda x: int(x.split("_")[-1]))

    ckpt_deleted = []
    for ckpt_file in ckpt_files[:-topk]:
        fs.rm(ckpt_file, recursive=True)
        ckpt_deleted.append(ckpt_file)
    return ckpt_deleted


def check_checkpoint_path_access(checkpoint_path: str, rank: int, world_rank_hv: int | None = None):
    if world_rank_hv:
        dummy_file_path = os.path.join(
            checkpoint_path, get_diloco_rank_dir_name(world_rank_hv), f"dummy_file_{rank}.txt"
        )
    else:
        dummy_file_path = os.path.join(checkpoint_path, f"dummy_file_{rank}.txt")

    with fsspec.open(dummy_file_path, "w") as f:
        f.write("This is a dummy file for testing access.")
    gfs = GenericFileSystem()
    gfs.rm(dummy_file_path)


def get_diloco_rank_dir_name(world_rank_diloco: int) -> str:
    return f"diloco_rank_{world_rank_diloco}"
