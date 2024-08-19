import pickle
import subprocess
import numpy as np
import pytest
import socket
from hivemind.dht.dht import DHT
from open_diloco.ckpt_utils import CKPT_PREFIX


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
        "--metric_logger_type",
        "dummy",
    ]


@pytest.mark.parametrize("num_gpu", [2])
def test_multi_gpu_ckpt(config, random_available_port, num_gpu, tmp_path):
    ckpt_path = f"{tmp_path}/ckpt"
    log_file_1 = f"{tmp_path}/log1.json"
    log_file_2 = f"{tmp_path}/log2.json"

    run_1 = ["--ckpt.path", ckpt_path, "--ckpt.interval", "10", "--project", log_file_1]

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpu}",
        "--rdzv-endpoint",
        f"localhost:{random_available_port}",
        "open_diloco/train_fsdp.py",
        *config,
    ]

    result = subprocess.run(cmd + run_1)

    if result.returncode != 0:
        pytest.fail(f"Process {result} failed {result.stderr}")

    run_2 = ["--ckpt.path", ckpt_path, "--ckpt.resume", f"{ckpt_path}/{CKPT_PREFIX}_20", "--project", log_file_2]

    results_resume = subprocess.run(cmd + run_2)

    if results_resume.returncode != 0:
        pytest.fail(f"Process {result} failed {result.stderr}")

    with open(log_file_1, "rb") as f:
        log1 = pickle.load(f)
    with open(log_file_2, "rb") as f:
        log2 = pickle.load(f)

    log1 = {data["step"]: [data["Loss"], data["lr"]] for data in log1}
    log2 = {data["step"]: [data["Loss"], data["lr"]] for data in log2}

    common_step = set(log1.keys()) & set(log2.keys())

    for step in common_step:
        assert np.allclose(log1[step][0], log2[step][0], atol=1e-3), f"Loss at step {step} is different"
        assert log1[step][1] == log2[step][1], f"Lr at step {step} is different"


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
        "--metric_logger_type",
        "dummy",
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
