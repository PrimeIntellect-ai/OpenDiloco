import pytest
import os
from unittest import mock
import socket
from contextlib import contextmanager


from open_diloco.fsdp_diloco_native import Config, ddp_setup, destroy_process_group
from open_diloco.fsdp_diloco_native import train


def get_random_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def random_available_port():
    return get_random_available_port()


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


@pytest.fixture
def config() -> Config:
    return Config(
        lr=1e-2,
        total_batch_size=32,
        per_device_train_batch_size=8,
        max_steps=10,
    )


def test_fsdp_diloco_native(simple_ddp_environment, config):
    train(config)
