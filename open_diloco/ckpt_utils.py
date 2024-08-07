import fsspec
import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.distributed.checkpoint as dcp
import os
from torchdata.stateful_dataloader import StatefulDataLoader

GLOBAL_STATE_FILE = "global_state_dict.pt"


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
    
    if outer_optimizer is not None:
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
    else:
        model_state_dict, optimizer_state_dict = get_state_dict(model, [optimizer, outer_optimizer])

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

    optimizers = [optimizer] 
    if outer_optimizer is not None:
        optimizers.append(outer_optimizer)

    model_state_dict, optimizers_state_dict = get_state_dict(model, optimizers)

    dcp_state_dict = {
        "model": model_state_dict,
        "optimizers": optimizers_state_dict,
    }
    dcp.load(dcp_state_dict, storage_reader=fs_storage_reader)

    if outer_optimizer is not None:
        set_state_dict(
            model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizers_state_dict,
        )
    else:
        set_state_dict(
            model,
            [optimizer, outer_optimizer],
            model_state_dict=model_state_dict,
            optim_state_dict=optimizers_state_dict,
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
