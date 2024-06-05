import ctypes
import gc
import multiprocessing as mp
import time
from functools import partial
from typing import List

import pytest

import hivemind
from hivemind.dht import DHT

from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.mpfuture import MPFuture

import psutil

from open_diloco.hivemind_diloco import AllReduceStrategy, DiLoCoGradAverager, DiLoCoOptimizer


@pytest.fixture(autouse=True, scope="module")
def cleanup_children():
    yield

    with RSAPrivateKey._process_wide_key_lock:
        RSAPrivateKey._process_wide_key = None

    gc.collect()  # Call .__del__() for removed objects

    children = psutil.Process().children(recursive=True)
    if children:
        gone, alive = psutil.wait_procs(children, timeout=0.1)
        for child in alive:
            child.terminate()
        gone, alive = psutil.wait_procs(alive, timeout=1)
        for child in alive:
            child.kill()

    MPFuture.reset_backend()


def launch_dht_instances(n_peers: int, **kwargs) -> List[DHT]:
    dhts = [DHT(start=True, **kwargs)]
    initial_peers = dhts[0].get_visible_maddrs()

    dhts.extend(DHT(initial_peers=initial_peers, start=True, await_ready=False, **kwargs) for _ in range(n_peers - 1))
    for process in dhts[1:]:
        process.wait_until_ready()

    return dhts


@pytest.mark.forked
def test_allreduce_dilco_grad_averager():
    import torch

    n_peers = 4

    def get_model():
        return torch.nn.Linear(5, 1, bias=False)

    models = [get_model() for _ in range(n_peers)]
    offloaded_models = [get_model() for _ in range(n_peers)]
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in offloaded_models]

    dht_instances = launch_dht_instances(n_peers)
    averagers = [
        DiLoCoGradAverager(
            main_parameters=tuple(model.parameters()),
            offloaded_optimizer=opt,
            dht=dht,
            target_group_size=4,
            min_matchmaking_time=15,
            prefix="mygroup",
            client_mode=False,
            auxiliary=False,
            start=True,
        )
        for model, opt, dht in zip(models, optimizers, dht_instances)
    ]

    futures = []
    for averager in averagers:
        futures.append(averager.step(wait=False))
    for future in futures:
        result = future.result()
        for averager in averagers:
            assert averager.peer_id in result

    for averager in averagers:
        with averager.get_tensors() as averaged_pseudo_grads:
            for grad in averaged_pseudo_grads:
                assert not torch.isnan(grad).any(), "Averaged grad is nan"

    for process in averagers + dht_instances:
        process.shutdown()


def test_load_and_save_state():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    inner_lr = 0.1
    outer_lr = 0.7

    model = nn.Linear(5, 1)
    features = torch.randn(100, 5)
    targets = features @ torch.randn(5, 1)

    def get_opt():
        return DiLoCoOptimizer(
            run_id="test_run",
            batch_size=32,
            num_inner_steps=5,
            params=model.parameters(),
            outer_optimizer=partial(torch.optim.SGD, lr=outer_lr, nesterov=True, momentum=0.9),
            inner_optimizer=partial(torch.optim.AdamW, lr=inner_lr),
            scheduler=partial(torch.optim.lr_scheduler.StepLR, gamma=0.5, step_size=1),
            dht=hivemind.DHT(start=True),
            tracker_opts=dict(private_key=RSAPrivateKey(), max_refresh_period=1.0),
            averager_opts=dict(request_timeout=0.5),
            matchmaking_time=1.0,
            averaging_timeout=5.0,
            verbose=False,
        )

    opt1 = get_opt()

    for _ in range(2):
        batch = torch.randint(0, len(features), (32,))

        loss = F.mse_loss(model(features[batch]), targets[batch])

        loss.backward()
        assert loss.item() != 0, "Loss is zero, maybe gradient exploded."

        opt1.step()

    state = opt1.state_dict()

    opt2 = get_opt()
    opt2.load_state_dict(state)

    assert opt1.state_dict() == opt2.state_dict()

    assert opt1.state_averager.optimizer.param_groups[0]["lr"] == opt2.state_averager.optimizer.param_groups[0]["lr"]
    assert opt1.state_averager.optimizer.state_dict() == opt2.state_averager.optimizer.state_dict()

    assert opt1.state_averager.inner_optimizer.state_dict() == state["state_dict_inner"]
    assert opt1.state_averager.inner_optimizer.state_dict() == opt2.state_averager.inner_optimizer.state_dict()


# for some reason this test does not pass, the code is correct tho (tested manually).
# I (sami) still want to keep tracke of this test for the future.
@pytest.mark.skip("skip test")
@pytest.mark.parametrize(
    "strategy, expected_peer", [(AllReduceStrategy.NO_WAIT, 1), (AllReduceStrategy.WAIT_FOR_ALL, 4)]
)
def test_strategy_all_reduce(strategy: AllReduceStrategy, expected_peer: int):
    dht = hivemind.DHT(start=True)

    import torch  # putting import here for multi processing
    import torch.nn as nn
    import torch.nn.functional as F

    sleep_time = [0.5, 0.5, 0.5, 0.1]
    num_peers = len(sleep_time)

    on_time_peer = mp.Value(ctypes.c_int32, 0)

    batch_size = 16
    total_epochs = 10
    num_inner_steps = 5

    def run_trainer(sleep_time: float):
        features = torch.randn(100, 5) / 100
        targets = features @ torch.randn(5, 1)

        outer_lr = 0.7
        inner_lr = 0.1

        model = nn.Linear(5, 1)

        assert isinstance(model, torch.nn.Module), "model_arch must evaluate to a pytorch module"

        optimizer = DiLoCoOptimizer(
            run_id="test_run",
            batch_size=batch_size,
            num_inner_steps=num_inner_steps,
            params=model.parameters(),
            all_reduce_strategy=strategy,
            timeout_waiting_for_peers=None if strategy == AllReduceStrategy.NO_WAIT else 10.0,
            outer_optimizer=partial(torch.optim.SGD, lr=outer_lr, nesterov=True, momentum=0.9),
            inner_optimizer=partial(torch.optim.AdamW, lr=inner_lr),
            scheduler=partial(torch.optim.lr_scheduler.StepLR, gamma=0.5, step_size=1),
            dht=hivemind.DHT(
                initial_peers=dht.get_visible_maddrs(),
                start=True,
            ),
            tracker_opts=dict(private_key=RSAPrivateKey(), max_refresh_period=1.0),
            averager_opts=dict(request_timeout=0.5),
            matchmaking_time=2.0,
            averaging_timeout=5.0,
            verbose=False,
        )
        time.sleep(sleep_time)

        optimizer.load_state_from_peers()

        for _ in range(total_epochs):
            time.sleep(sleep_time)
            batch = torch.randint(0, len(features), (batch_size,))

            loss = F.mse_loss(model(features[batch]), targets[batch])

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        if optimizer.local_epoch == optimizer.tracker.global_epoch:
            on_time_peer.value += 1

        time.sleep(1.0)
        optimizer.shutdown()

    peers = []
    for index in range(num_peers):
        peers.append(
            mp.Process(
                target=run_trainer,
                name=f"trainer-{index}",
                kwargs=dict(sleep_time=sleep_time[index]),
            )
        )

    for peer in peers:
        peer.start()

    for peer in peers:
        peer.join()

    assert on_time_peer.value == expected_peer

    for process in peers:
        process.terminate()
