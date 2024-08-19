import subprocess
import pytest
import socket
import os
from unittest import mock
from hivemind.dht.dht import DHT


@pytest.fixture(autouse=True)
def set_env():
    os.environ["WANDB_MODE"] = "disabled"

    with mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
        yield


def get_random_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def random_available_port():
    return get_random_available_port()


@pytest.fixture
def config() -> list[str]:
    return [
        "--path_model",
        "tests/models/llama-2m-fresh",
        "--fake_data",
        "--no-torch_compile",
        "--lr",
        "1e-2",
        "--per_device_train_batch_size",
        "8",
        "--total_batch_size",
        "16",
        "--max_steps",
        "50",
    ]


@pytest.mark.parametrize("num_gpu", [1, 2])
def test_multi_gpu(config, random_available_port, num_gpu):
    result = subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={num_gpu}",
            "--rdzv-endpoint",
            f"localhost:{random_available_port}",
            "open_diloco/train_fsdp.py",
            *config,
        ],
    )

    if result.returncode != 0:
        pytest.fail(f"Process {result} failed {result.stderr}")


@pytest.fixture
def config_hv() -> list[str]:
    config = [
        "--path_model",
        "tests/models/llama-2m-fresh",
        "--fake_data",
        "--no-torch_compile",
        "--lr",
        "1e-2",
        "--per_device_train_batch_size",
        "8",
        "--total_batch_size",
        "16",
        "--max_steps",
        "100",
    ]

    return config + [
        "--hv.local_steps",
        "25",
        "--hv.skip_load_from_peers",
        "--hv.fail_rank_drop",
        "--hv.matchmaking_time",
        "5",
    ]


@pytest.mark.parametrize("num_diloco", [1, 2])
def test_multi_gpu_hivemind(config_hv, num_diloco):
    dht = DHT(
        start=True,
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{get_random_available_port()}"],
    )

    initial_peers = str(dht.get_visible_maddrs()[0])

    results = []

    for i in range(num_diloco):
        port = get_random_available_port()
        result = subprocess.Popen(
            [
                "torchrun",
                f"--nproc_per_node={1}",
                "--rdzv-endpoint",
                f"localhost:{port}",
                "open_diloco/train_fsdp.py",
                *config_hv,
                "--hv.initial_peers",
                initial_peers,
                "--hv.world_rank",
                str(i),
                "--hv.galaxy_size",
                str(num_diloco),
            ],
        )
        results.append(result)

    for result in results:
        result.wait()
        if result.returncode != 0:
            pytest.fail(f"Process {result} failed {result.stderr}")
