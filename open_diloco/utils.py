import hashlib
from functools import partial
from typing import Any, Generator

import torch
from torch.utils.hooks import RemovableHandle
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import IterableDataset


_FSDP_WRAPPED_MODULE = ["_forward_module.", "_fsdp_wrapped_module."]


def _remove_fsdp_prefix(name: str) -> str:
    for prefix in _FSDP_WRAPPED_MODULE:
        if prefix in name:
            return name.replace(prefix, "")
    return name


@torch.no_grad()
def log_activations_hook(
    _mod: torch.nn.Module,
    _inp: torch.Tensor,
    outp: torch.Tensor | tuple[torch.Tensor, ...],
    mod_name: str,
    log_activations: dict[str, float],
) -> None:
    if isinstance(outp, tuple):
        outp = outp[0]

    norm = outp.norm(p=2)

    name = _remove_fsdp_prefix(mod_name)

    if f"activation/{name}" not in log_activations:
        log_activations[f"activation/{name}"] = norm
    else:
        log_activations[f"activation/{name}"] += norm


class ActivationNormMetric:
    """
    This class is used to monitor the norm of the activation of the target layers.
    It attached hook to the forward of each layer that will log the output, and remove them after.
    """

    def __init__(self, target_layers: list[str], gradient_accumulation_steps: int):
        self.target_layers = target_layers
        self.handles: list[RemovableHandle] = []
        self._log_activations: dict[str, torch.Tensor] = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def register_metrics_hooks(self, model: torch.nn.Module):
        """
        this function take a torch module, a list of layer name and apply a hook function that
        monitor the output norm of the layers.
        """
        handles = []
        for name, mod in model.named_modules():
            for layer in self.target_layers:
                if name.endswith(layer):
                    handle = mod.register_forward_hook(
                        partial(log_activations_hook, log_activations=self._log_activations, mod_name=name)
                    )
                    handles.append(handle)
                    break

        self.handles = handles

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()

    @property
    def log_activations(self) -> dict[str, torch.Tensor]:
        return {k: v / self.gradient_accumulation_steps for k, v in self._log_activations.items()}


def _round_str(x: float):
    return f"{x:.4f}"


def _round_flatten(a: torch.Tensor, max_size: int = 1000):
    bounds = int(max_size**0.5)
    return ",".join(_round_str(i) for i, _ in zip(a[:bounds, :bounds].flatten(), range(max_size)))


def hash_tensor_content(a: torch.Tensor, max_size: int = 1000) -> str:
    return hashlib.md5(_round_flatten(a, max_size=max_size).encode("utf-8")).hexdigest()


def get_compression_kwargs(hivemind_compression: str | None) -> dict:
    """Return the compression kwargs for hivemind optimizer based on the hivemind_compression argument."""
    ret_kwargs = {}

    if hivemind_compression is None:
        from hivemind import NoCompression

        ret_kwargs["grad_compression"] = NoCompression()
        ret_kwargs["state_averaging_compression"] = NoCompression()

    elif hivemind_compression == "fp16":
        from hivemind import Float16Compression

        ret_kwargs["grad_compression"] = Float16Compression()
        ret_kwargs["state_averaging_compression"] = Float16Compression()
    elif hivemind_compression == "scaled-fp16":
        from hivemind import ScaledFloat16Compression

        ret_kwargs["grad_compression"] = ScaledFloat16Compression()
        ret_kwargs["state_averaging_compression"] = ScaledFloat16Compression()
    elif hivemind_compression == "uniform8bit":
        from hivemind import Uniform8BitQuantization

        ret_kwargs["grad_compression"] = Uniform8BitQuantization()
        ret_kwargs["state_averaging_compression"] = Uniform8BitQuantization()
    elif hivemind_compression == "quantile8bit":
        from hivemind import Quantile8BitQuantization

        ret_kwargs["grad_compression"] = Quantile8BitQuantization()
        ret_kwargs["state_averaging_compression"] = Quantile8BitQuantization()

    elif hivemind_compression == "blockwise8bit":
        from hivemind import BlockwiseQuantization

        ret_kwargs["grad_compression"] = BlockwiseQuantization()
        ret_kwargs["state_averaging_compression"] = BlockwiseQuantization()
    else:
        raise ValueError(f"Invalid hivemind_compression: {hivemind_compression}")
    return ret_kwargs


def found_inf_grad(optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler) -> bool:
    """
    this function check if the scaler has found inf grad for the optimizer. It does by looking up the optimizer state
    regsited inside the scaler. Code is mostly copied/inspired by the torch GradScaler codebase.
    """
    if not scaler._enabled:
        return False

    optimizer_state = scaler._per_optimizer_states[id(optimizer)]
    assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."

    return sum(v.item() for v in optimizer_state["found_inf_per_device"].values()) > 0


def get_sharding_strategy(sharding_strategy: str) -> ShardingStrategy:
    if sharding_strategy == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    elif sharding_strategy == "HYBRID_SHARD":
        return ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "_HYBRID_SHARD_ZERO2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Please choose 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD', or '_HYBRID_SHARD_ZERO2'."
        )


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            input_ids = torch.randint(3, self.vocab_size, (self.seq_len,)).tolist()
            yield {"input_ids": input_ids}


def collate_causal_mask(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100) -> callable:
    return partial(_collate_fn_causal_mask, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)


def _collate_fn_causal_mask(
    samples: list[dict[str, torch.LongTensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
) -> dict[str, torch.LongTensor]:
    assert samples[0].keys() == {"input_ids"}

    batched = {"input_ids": [], "labels": []}

    if max_seq_length > 0:
        max_seq_length += 1  # this makes sure that the effective seqlen is correct

    for sample in samples:
        input_ids = torch.Tensor(sample["input_ids"]).long()

        if len(input_ids) < max_seq_length:
            input_ids = torch.cat([input_ids, torch.full((max_seq_length - len(input_ids),), pad_id)])
        elif len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        batched["input_ids"].append(input_ids[1:])
        batched["labels"].append(input_ids[:-1])

    return {"input_ids": torch.stack(batched["input_ids"], dim=0), "labels": torch.stack(batched["labels"], dim=0)}
