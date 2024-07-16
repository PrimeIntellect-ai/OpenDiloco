import pytest
import os
from unittest import mock
import socket
from contextlib import contextmanager
import multiprocessing
import copy
import torch
import gc

from hivemind.dht.dht import DHT
from open_diloco.train_fsdp import train, Config, ddp_setup, destroy_process_group, HvConfig
from open_diloco.llama import Config as ModelConfig


@pytest.fixture(autouse=True)
def set_env():
    os.environ["WANDB_MODE"] = "disabled"

    with mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
        yield


@pytest.fixture(autouse=True)
def memory_cleanup():
    # credits to : https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117
    try:
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def get_random_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def random_available_port():
    return get_random_available_port()


@pytest.fixture
def config() -> Config:
    model_config = ModelConfig(
        name="llama_2m",
        n_embd=64,
        intermediate_size=256,
        n_head=2,
        n_layer=2,
        vocab_size=1024,
    )
    return Config(
        llama_config=model_config,
        fake_data=True,
        torch_compile=False,
        lr=1e-2,
        per_device_train_batch_size=8,
        total_batch_size=16,
        max_steps=10,
    )


@contextmanager
def ddp_environment(random_available_port, local_rank=0, world_size=1):
    with mock.patch.dict(
        os.environ,
        {
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "RANK": str(local_rank),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(random_available_port),
        },
    ):
        ddp_setup()
        try:
            yield
        finally:
            destroy_process_group()


@pytest.fixture
def simple_ddp_environment(random_available_port):
    with ddp_environment(random_available_port, local_rank=0, world_size=1):
        yield


def test_train(config, simple_ddp_environment):
    train(config)


@pytest.mark.parametrize("world_size", [2])
def test_multi_gpu(config, random_available_port, world_size):
    def worker(local_rank):
        with ddp_environment(random_available_port, local_rank=local_rank, world_size=world_size):
            train(config)

    processes = [multiprocessing.Process(target=worker, args=(rank,)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


@pytest.fixture
def diloco_config(config: Config) -> Config:
    hv_config = HvConfig(local_steps=5, skip_load_from_peers=True, world_rank=0, galaxy_size=1)
    config.hv = hv_config

    return config


@pytest.mark.parametrize("galaxy_size", [2])
def test_diloco_train(diloco_config: Config, galaxy_size):
    dht = DHT(start=True)
    diloco_config.hv.initial_peers = dht.get_visible_maddrs()
    diloco_config.max_steps = 100

    def worker(world_rank):
        with ddp_environment(get_random_available_port(), local_rank=0, world_size=1):
            config_copy: Config = copy.deepcopy(diloco_config)
            config_copy.hv.galaxy_size = galaxy_size
            config_copy.hv.world_rank = world_rank
            train(config_copy)

    processes = [multiprocessing.Process(target=worker, args=(rank,)) for rank in range(galaxy_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
