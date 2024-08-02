from enum import Enum
import time
from typing import Callable, Iterator, List, Optional, Union
import numpy as np

import torch

from hivemind.averaging.averager import DecentralizedAverager
from hivemind.averaging.control import StepControl
from hivemind.compression.base import CompressionBase, NoCompression
from hivemind.dht.dht import DHT
from hivemind.optim.optimizer import Optimizer
from hivemind.optim.progress_tracker import (
    GlobalTrainingProgress,
    ProgressTracker,
    TrainingProgressSchema,
)
from hivemind.optim.state_averager import (
    LRSchedulerBase,
    OptimizerFactory,
    Parameters,
    ParamGroups,
    SchedulerFactory,
    TorchOptimizer,
    TrainingStateAverager,
)
from hivemind.utils import get_dht_time
from hivemind.utils.timed_storage import DHTExpiration
from hivemind.optim.optimizer import logger
from hivemind.optim.progress_tracker import LocalTrainingProgress

from open_diloco.utils import found_inf_grad


class DiLoCoStateAverager(TrainingStateAverager):
    def __init__(
        self,
        *,
        num_inner_steps: int,
        inner_optimizer: TorchOptimizer,
        scheduler: Optional[SchedulerFactory] = None,
        **kwargs,
    ):
        self.inner_optimizer = inner_optimizer
        self.num_inner_steps = num_inner_steps

        super().__init__(
            **kwargs
        )  # we specifically don't pass the scheduler here, default TrainingStateAverager would use it with the outer optimizer and we w

        self.scheduler_inner_optimizer = scheduler(self.inner_optimizer) if scheduler is not None else None
        assert isinstance(self.scheduler_inner_optimizer, (LRSchedulerBase, type(None)))

    def _update_scheduler(self):
        """Increase the scheduler state until it becomes synchronized with local epoch"""
        # TODO(sami) handle update scheduler
        # for now assuming that all scheduler are on time
        pass


class DiLoCoGradAverager(DecentralizedAverager):
    """ "
    DiLoCoGradAverager is meant to be used in pair with DiLoCoStateAverager. Specifically it takes as input the offloaded optimizer of DiLoCoStateAverager, and
    use the grad buffer of the offloaded param as averaged_tensors for the DecentralizedAverager. In other words the DiLoCoGradAverager makes sure that the grad of the offloaded optimizer
    are kept in sync between peers.
    """

    def __init__(
        self,
        main_parameters: List[torch.nn.Parameter],
        offloaded_optimizer: TorchOptimizer,
        *,
        dht: DHT,
        prefix: str,
        warn: bool = True,
        **kwargs,
    ):
        if "client_mode" in kwargs:
            if kwargs["client_mode"] is not None and kwargs["client_mode"]:
                raise KeyError("client_mode is not supported in DiLoCoGradAverager")
            else:
                kwargs.pop("client_mode")

        if "averaged_grads" in kwargs:
            raise KeyError(
                "DiLoCoGradAverager does not support averaged_grads since it use the offloaded optimizer gradients directly"
            )

        if not isinstance(main_parameters, (list, tuple)):
            raise ValueError(
                "main_parameters must be a list or tuple of torch.nn.Parameter and not an iterator otherwise parameters will be consumed"
            )
        self.main_parameters = list(main_parameters)
        self.offloaded_optimizer = offloaded_optimizer

        self.warn = warn
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0

        self._new_averaged_grads = False

        averaged_grads = tuple(grad for grad in self._grads_from_optimizer())

        super().__init__(
            averaged_tensors=averaged_grads,
            dht=dht,
            prefix=prefix,
            client_mode=False,
            **kwargs,
        )

    def _grads_from_optimizer(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated optimizer"""
        param_groups = self.offloaded_optimizer.param_groups
        for param_group in param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                yield param.grad

    def schedule_step(self, scheduled_time: Optional[DHTExpiration] = None, **kwargs) -> StepControl:
        """
        Begin matchmaking: look for a group of peers and prepare for averaging gradients at a specified time.

        :param scheduled_time: expected time when to perform all-reduce. Can be changed using control.scheduled_time
        :param kwargs: any additional keyword args from DecentralizedAverager.step, such as gather, allow_retries, etc
        :note: setting weight at this stage is not supported, please leave this parameter as None
        :returns: step_control - a handle that can be passed into GradientAverager.step to use the pre-scheduled group
        :note: in the current implementation, each step_control can only be used in one step.
        """
        assert kwargs.get("weight") is None, "setting weight in schedule_step is not supported"
        return super().step(scheduled_time=scheduled_time, wait=False, require_trigger=True, **kwargs)

    def step(
        self,
        control: Optional[StepControl] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        **kwargs,
    ):
        """
        Average accumulated gradients with peers, optionally load averaged gradients and reset accumulators

        :param weight: overrides the averaging weight; by default, weight equals the number of accumulated samples
        :param reset_accumulators: by default, set local gradient accumulators to zeros after averaging succeeds
        :param control: reuse a pre-arranged group of peers (or a matchmaking in progress) from averager.schedule_step
        :param timeout: if specified, await for averaging round for at most this number of seconds (if wait=True)
        :param wait: if True, await for the step to finish (or fail), otherwise run all-reduce in background
        """
        if control is None:
            control = self.schedule_step(timeout=timeout, **kwargs)

        self.compute_and_load_pseudo_grad_into_averager()
        control.allow_allreduce()

        return control.result(timeout) if wait else control

    @torch.no_grad()
    def compute_and_load_pseudo_grad_into_averager(self):
        """compute pseudo gradient by subtracting the offloaded optimizer parameters with the main parameters and load them in the averager"""
        opt_parameters = [param for group in self.offloaded_optimizer.param_groups for param in group["params"]]
        with self.get_tensors() as averaged_grads:
            for opt_param, averaged_grad, main_param in zip(opt_parameters, averaged_grads, self.main_parameters):
                # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                grad = opt_param.data - main_param.detach().to(opt_param.device)
                averaged_grad.copy_(grad, non_blocking=True)

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False


class DiloCoProgressTracker(ProgressTracker):
    global_progress: GlobalTrainingProgress
    local_progress: LocalTrainingProgress

    def __init__(self, batch_size: int, num_inner_steps: int, **kwargs):
        self.batch_size = batch_size
        self.num_inner_steps = num_inner_steps
        super().__init__(**kwargs)

    @property
    def ready_to_update_epoch(self) -> bool:
        """Whether or not this peer can increment epoch right away."""
        return (
            self.global_epoch > self.local_progress.epoch
            or self.local_progress.samples_accumulated
            >= self.target_batch_size  # here we track local progress as each diloco worker need to do num_inner_steps (for now)
            # or get_dht_time() >= self.global_progress.eta_next_epoch # disabled for our test
        )

    @property
    def estimated_next_update_time(self) -> DHTExpiration:
        """Estimate (absolute) time when this peer should increment epoch"""
        if self.ready_to_update_epoch:
            return get_dht_time()

        samples_remaining_to_next_epoch = max(0, self.target_batch_size - self.local_progress.samples_accumulated)
        return samples_remaining_to_next_epoch / self.performance_ema.samples_per_second

    @property
    def local_step(self) -> int:
        return self.local_progress.samples_accumulated // self.batch_size

    @property
    def real_step(self) -> int:
        return self.local_step + self.local_progress.epoch * self.batch_size

    def _parse_swarm_progress_data(self, metadata: TrainingProgressSchema) -> GlobalTrainingProgress:
        """Read performance statistics reported by peers, estimate progress towards next batch
        This function is copy paste from hivemind. Only difference is that if fix the ETA estimation.
        """
        current_time = get_dht_time()

        if not isinstance(metadata, dict) or len(metadata) == 0:
            logger.log(self.status_loglevel, f"Found no active peers: {metadata}")
            samples_remaining_to_next_epoch = max(0, self.target_batch_size - self.local_progress.samples_accumulated)
            local_eta_next_epoch = samples_remaining_to_next_epoch / self.performance_ema.samples_per_second

            return GlobalTrainingProgress(
                self.local_progress.epoch,
                self.local_progress.samples_accumulated,
                self.target_batch_size,
                num_peers=0,
                num_clients=0,
                eta_next_epoch=current_time + local_eta_next_epoch,
                next_fetch_time=current_time + self.default_refresh_period,
            )

        valid_peer_entries = [
            LocalTrainingProgress.parse_obj(peer_state.value)
            for peer_state in metadata.values()
            if peer_state.value is not None
        ]

        num_peers = len(valid_peer_entries)
        num_clients = sum(peer.client_mode for peer in valid_peer_entries)

        global_epoch = self.local_progress.epoch
        for peer in valid_peer_entries:
            if not peer.client_mode:
                global_epoch = max(global_epoch, peer.epoch)

        total_samples_accumulated = 0
        total_samples_per_second = self.performance_ema.eps

        estimated_time_to_next_epoch = 0

        for peer in valid_peer_entries:
            total_samples_per_second += peer.samples_per_second
            if peer.epoch == global_epoch:
                samples_remaining_to_next_epoch = max(0, self.target_batch_size - peer.samples_accumulated)
                local_eta_next_epoch = samples_remaining_to_next_epoch / peer.samples_per_second

                estimated_time_to_next_epoch = max(estimated_time_to_next_epoch, local_eta_next_epoch)

            # note: we deliberately count only valid peers for samples_accumulated, but all peers for performance;
            # the rationale behind this is that outdated peers will synchronize and begin contributing shortly.

        time_to_next_fetch = float(
            np.clip(
                a=estimated_time_to_next_epoch,
                a_min=self.min_refresh_period,
                a_max=self.max_refresh_period,
            )
        )

        logger.log(
            self.status_loglevel,
            f"{self.prefix} has taken {self.local_step} local steps. Peers: {num_peers}, epoch: {self.local_progress.epoch}, steps: {self.real_step}. ETA: {estimated_time_to_next_epoch:.2f}",
        )

        return GlobalTrainingProgress(
            global_epoch,
            total_samples_accumulated,
            target_batch_size=self.target_batch_size,
            num_peers=num_peers,
            num_clients=num_clients,
            eta_next_epoch=current_time + estimated_time_to_next_epoch,
            next_fetch_time=current_time + time_to_next_fetch,
        )


class AllReduceStrategy(Enum):
    """
    DiLoCo support multiple strategy to trigger the pseudo gradient averaging step.

    stregy:
        * WAIT_FOR_ALL: DiLoCo will wait for all peers to finish their local updates before triggering the all reduce step
            use this strategy when you trust all of your peers
        * NO_WAIT: The fastest peer will trigger the all reduce as soon as it reach its local steps (modulo the amount of time it need to wait because of the `matchmaking_time`)
            use this strategy when some of your peers are unreliable
    """

    WAIT_FOR_ALL = "WAIT_FOR_ALL"
    NO_WAIT = "NO_WAIT"


DEFAULT_TIMEOUT_WAITING_FOR_PEERS = 600


class DiLoCoOptimizer(Optimizer):
    """
    DiLoCo optimizer extend Hivemind's Optimizer to support DiLoCo training with local updates, requiring less bandwidth to train
    and still converge.

    Pseudo gradient is the difference between the weight before and after the multiple local update of the inner optimizer.

    Paper:  https://arxiv.org/abs/2311.08105

    :param: outer_optimizer: Callable to an optimizer to update the pseudo gradient, this optimizer is shared between peers. (DiLoCo used the Nesterov opt)
    :param: inner_optimizer: Callable to an optimizer to update the model parameter locally, this optimizer is not shared between peers (DiLoCo used the AdamW opt)
    :param: scheduler: callable to a learning rate scheduler to update the inner optimizer lr.
    :param: num_inner_steps: number of inner optimizer updates per outer optimizer update
    :param: batch_size: number of samples in a single batch

    the rest of parameters are the same as Hivemind's Optimizer, expect `optimizer` that is override by `outer_optimizer`.
    """

    state_averager: DiLoCoStateAverager
    inner_optimizer: TorchOptimizer
    tracker: DiloCoProgressTracker
    diloco_grad_averager: DiLoCoGradAverager

    def __init__(
        self,
        *,
        dht: DHT,
        run_id: str,
        batch_size: int,
        num_inner_steps: int,
        outer_optimizer: OptimizerFactory,
        inner_optimizer: OptimizerFactory,
        params: Optional[Union[Parameters, ParamGroups]] = None,
        scheduler: Optional[SchedulerFactory] = None,
        outer_scheduler: Optional[SchedulerFactory] = None,
        averager_opts: Optional[dict] = None,
        grad_compression: CompressionBase = NoCompression(),
        tracker_opts: Optional[dict] = None,
        all_reduce_strategy: AllReduceStrategy = AllReduceStrategy.WAIT_FOR_ALL,
        timeout_waiting_for_peers: float | None = None,
        matchmaking_time: Optional[float] = 15.0,
        **kwargs,
    ):
        self._check_kwargs(kwargs)

        if timeout_waiting_for_peers is not None:
            if all_reduce_strategy == AllReduceStrategy.NO_WAIT:
                raise ValueError(
                    "You cannot use timeout_waiting_for_peers with NO_WAIT strategy, use WAIT_FOR_ALL instead"
                )

        if timeout_waiting_for_peers is not None and timeout_waiting_for_peers < matchmaking_time:
            raise ValueError("timeout_waiting_for_peers must be greater than matchmaking_time")

        if all_reduce_strategy == AllReduceStrategy.WAIT_FOR_ALL:
            if timeout_waiting_for_peers is None:
                timeout_waiting_for_peers = DEFAULT_TIMEOUT_WAITING_FOR_PEERS

        self.all_reduce_strategy = all_reduce_strategy
        self.timeout_waiting_for_peers = timeout_waiting_for_peers

        params = list(params)
        # if params is a generator (like model.parameters()) it would be consumed by the first optimizer
        # since we have two optimizers, we need to persist the params to a list
        self.num_inner_steps = num_inner_steps

        for opt_or_scheduler in [outer_optimizer, scheduler, outer_scheduler]:
            if not (callable(opt_or_scheduler) or opt_or_scheduler is None):
                raise TypeError("You need to pass inner and outer optimizer as well as scheduler as callable")

        if isinstance(inner_optimizer, TorchOptimizer):
            self.inner_optimizer = inner_optimizer
        elif isinstance(inner_optimizer, Callable):
            self.inner_optimizer = inner_optimizer(params=params)
        else:
            raise TypeError(
                f"Expected inner_optimizer to be TorchOptimizer or OptimizerFactory, got {type(inner_optimizer)}"
            )

        if tracker_opts is None:
            tracker_opts = {}

        tracker_opts.update(dict(batch_size=batch_size, num_inner_steps=num_inner_steps))

        if "max_refresh_period" not in tracker_opts:
            tracker_opts["max_refresh_period"] = 2

        self.scheduled_diloco_grads: Optional[StepControl] = None

        super().__init__(
            optimizer=outer_optimizer,
            dht=dht,
            run_id=run_id,
            target_batch_size=batch_size * num_inner_steps,
            batch_size_per_step=batch_size,
            params=params,
            scheduler=scheduler,
            use_local_updates=True,  # we are handling grad scaler ourself
            offload_optimizer=True,  # DiLoCo is always offloading optimizers bc of the pseudo gradient
            averager_opts=averager_opts,
            tracker_opts=tracker_opts,
            matchmaking_time=matchmaking_time,
            **kwargs,
        )
        self.diloco_grad_averager = self._make_gradient_averager(compression=grad_compression)

        self.outer_scheduler = outer_scheduler(self.state_averager.optimizer)

    def _check_kwargs(self, kwargs) -> None:
        """DiLoCo Optimizer only support a subset of Hivemind Optimizer kwargs.
        This function raise an error if some kwargs are not supported"""

        if "optimizer" in kwargs:
            raise KeyError("optimizer should not be passed to DiLoCoOptimizer, pass rather to outer_optimizer")

        if "use_local_updates" in kwargs:
            if kwargs["use_local_updates"] is False:
                raise ValueError(
                    "You cannot use DiLoCo without local updates, please use normal Optimizer if you don't want local updates"
                )
            else:
                kwargs.pop("use_local_updates")

        if "offload_optimizer" in kwargs:
            if kwargs["offload_optimizer"] is False:
                raise ValueError("offload_optimizer=False, is not supported in DiLoCo for now")
            else:
                kwargs.pop("offload_optimizer")

        for arg_name in (
            "delay_state_averaging",
            "delay_grad_averaging",
            "delay_optimizer_step",
        ):
            if arg_name in kwargs:
                if kwargs[arg_name] is True:
                    raise ValueError(f"{arg_name} is not supported in DiLoCo for now")

        if "target_batch_size" in kwargs:
            raise KeyError(
                "DiLoCo does not have a target_batch_size, use batch_size instead in combination with num_inner_steps"
            )

        if "batch_size_per_step" in kwargs:
            raise KeyError("DiLoCo does not have a batch_size_per_step, use batch_size instead")

    def _make_gradient_averager(self, **kwargs) -> DiLoCoGradAverager:
        assert hasattr(self, "state_averager"), "must initialize state averager first"
        grad_averager = DiLoCoGradAverager(
            dht=self.dht,
            prefix=f"{self.run_id}_grad_averager",
            main_parameters=self.state_averager.main_parameters,
            offloaded_optimizer=self.state_averager.optimizer,
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            **kwargs,
        )
        return grad_averager

    def _make_state_averager(self, **kwargs) -> DiLoCoStateAverager:
        return DiLoCoStateAverager(
            dht=self.dht,
            prefix=f"{self.run_id}_state_averager",
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            offload_optimizer=self.offload_optimizer,
            custom_gradients=self.offload_optimizer,
            status_loglevel=self.status_loglevel,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            start=True,
            num_inner_steps=self.num_inner_steps,
            inner_optimizer=self.inner_optimizer,
            **kwargs,
        )

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        """
        Note: code is is copied from Hivemind's Optimizer.step, the main change is that the local step is used with the **iner optimizer**, only
        the global step that sync data via all reduce is using the **outer optimizer** states.

        Note: There is no gradient accumulation in our DiLoCo implementation since we use local updates.

        Note2: the gradient scaler is only apply to the inner optimizer step. The outer optimizer step is working on pseudo gradient
        that don't need to be scaled.

        Note3: You should not call scaler.step(optimizer) but rather optimizer.step(scaler=scaler) otherwise the scaler will not work as expected because of the outer step.

        Update training. Depending on the configuration, this will
        report progress to peers, run global or local optimizer step, average parameters or schedule background tasks.

        Grad scaler must be pass to use mixed precision with the inner optimizer. One can call unscale_ before tho.

        :param closure: A closure that reevaluates the model and returns the loss.
        :param batch_size: optional override for batch_size_per_step from init.
        :param scaler: a scaler from torch.cuda.amp.GradScaler, if provided, the scaler will be used to scale the inner optimizer step but not the outer optimizer step.
        :note: this .step is different from normal pytorch optimizers in several key ways. See __init__ for details.
        """
        ### OG HIVEMIND CODE START ###
        if self.batch_size_per_step is None and batch_size is None and not self.auxiliary:
            raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
        if self.auxiliary and (closure is not None or batch_size is not None):
            raise ValueError("Auxiliary peers should not have batch size, run closures, or use grad_scaler")
        if scaler is not None and closure is not None:
            raise ValueError("You cannot use closure and scaler at the same time")

        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        # if delayed updates finished before step, apply these updates; otherwise do nothing
        # self.state_averager.step(apply_delayed_updates=True)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.auxiliary and self._should_load_state_from_peers():
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return loss  # local gradients were computed with out-of-sync parameters, must start over

        ### OG HIVEMIND CODE END ###

        # this code is similar to the hivemind.Optimizer.step when `use_local_updates` is True
        # at the difference that it call the inner optimizer step as well.

        if not self.auxiliary:
            new_samples_accumulated = self.tracker.local_progress.samples_accumulated + batch_size
            self.tracker.report_local_progress(self.local_epoch, samples_accumulated=new_samples_accumulated)

            self._maybe_schedule_state_averaging()
            self._maybe_schedule_gradient_averaging()

            if scaler is not None:
                scaler.step(self.inner_optimizer)
                if found_inf_grad(self.inner_optimizer, scaler):
                    logger.log(self.status_loglevel, f"Found inf grad at step {self.tracker.real_step}")
            else:
                self.inner_optimizer.step(closure=closure)

            if self.state_averager.scheduler_inner_optimizer:
                self.state_averager.scheduler_inner_optimizer.step()

        if self.tracker.ready_to_update_epoch:
            self._update_global_epoch()

        if self.outer_scheduler is not None:
            self.outer_scheduler.step()

        return loss

    def _compute_schema_hash(self) -> int:
        """this function is similar to hivemind.Optimizer._compute_schema_hash
        but disregard the gradient buffers of the offloaded optimizer
        """
        optimized_param_groups = self.state_averager.optimizer.param_groups
        optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
        param_shapes = tuple(tuple(param.shape) for param in optimized_parameters)
        grad_ids = None
        return hash((grad_ids, param_shapes))

    def _update_global_epoch(self) -> None:
        """Depending on the configuration: aggregate gradients and/or parameters, perform global optimizer step

        NOTE: this has been mostly copied from hivemind.Optimizer._update_global_epoch, except highlighted lines
        """
        assert self._schema_hash == self._compute_schema_hash(), "parameters changed during iteration"
        _epoch_start_time = time.perf_counter()

        if self.tracker.global_progress.num_peers > 1:
            if self.all_reduce_strategy == AllReduceStrategy.WAIT_FOR_ALL:
                if self.scheduled_diloco_grads is None:
                    init_time_waiting = time.perf_counter()

                    timeout_triggered = False

                    while time.perf_counter() - init_time_waiting < self.timeout_waiting_for_peers:
                        eta_next_epoch = self.tracker.global_progress.eta_next_epoch - get_dht_time()
                        if eta_next_epoch > self.matchmaking_time:
                            time_to_wait = max(0.1, self.tracker.global_progress.next_fetch_time - get_dht_time())
                            logger.log(
                                self.status_loglevel,
                                f"ETA next epoch {eta_next_epoch}, refresh in {time_to_wait}",
                            )
                            time.sleep(time_to_wait)
                        else:
                            logger.log(
                                self.status_loglevel,
                                f"Pre-scheduling gradient averaging round in {self.matchmaking_time:.2f} sec",
                            )
                            break
                    else:
                        timeout_triggered = True

                    if timeout_triggered:
                        logger.log(
                            self.status_loglevel,
                            "Timeout waiting for peers all-reduce was triggered. Going to skip slowest peers",
                        )
                        # todo(sami) in this case we still will have to wait for min_matchmaking_time, this could be optimized

        with self.tracker.pause_updates():
            assert not self.delay_optimizer_step, "delay_optimizer_step must be False in DiLoCo"

            if self.tracker.global_progress.num_peers > 1:
                logger.log(self.status_loglevel, f"Beginning optimizer step #{self.local_epoch}")
                time_0 = time.perf_counter()

                self.diloco_grad_averager.step(
                    wait=True, timeout=self.averaging_timeout, control=self.scheduled_diloco_grads
                )
                time_1 = time.perf_counter()
                logger.log(
                    self.status_loglevel,
                    f"Time taken for gradient all reduce: {time_1 - time_0} sec",
                )

                self.diloco_grad_averager.notify_used_averaged_gradients()
                self.scheduled_diloco_grads = None
            else:
                self.diloco_grad_averager.compute_and_load_pseudo_grad_into_averager()

            next_epoch = max(self.local_epoch + 1, self.tracker.global_epoch)
            swarm_not_empty = self.tracker.global_progress.num_peers > 1
            should_perform_optimizer_step = True  # different from hivemind.Optimizer
            should_average_state = (
                swarm_not_empty
                and next_epoch % self.average_state_every == 0
                and not self.state_averager.averaging_in_progress
            )

            if should_average_state and self.scheduled_state is not None:
                if self.scheduled_state.triggered or self.scheduled_state.done():
                    logger.log(
                        self.status_loglevel,
                        f"Not using pre-scheduled group for state averaging because it"
                        f"was already used elsewhere: {self.scheduled_state}",
                    )
                    self.scheduled_state = None
                self.delay_before_state_averaging.update(task_size=1, interval=time.perf_counter() - _epoch_start_time)

            assert self.state_averager.custom_gradients, "custom gradient must be enable for syncing pseudo gradients"

            logger.info(f"Try outer optimizer step at  {self.tracker.real_step} step")

            self.state_averager.step(
                increment_epoch=True,
                wait_for_trigger=None,
                optimizer_step=should_perform_optimizer_step,
                delay_optimizer_step=self.delay_optimizer_step and should_perform_optimizer_step,
                grad_scaler=None,
                averaging_round=should_average_state,
                delay_averaging=self.delay_state_averaging and not self.auxiliary,
                averaging_control=(self.scheduled_state if should_average_state else None),
                averaging_opts=(dict(timeout=self.averaging_timeout) if should_average_state else None),
                zero_grad=False,  # zero grad should be done outside of diloco
            )

            if not should_average_state and self.scheduled_state is not None and not self.scheduled_state.done():
                self.scheduled_state.cancel()
            self.scheduled_state = None

            self.tracker.update_epoch(new_epoch=self.state_averager.local_epoch)
            self._should_check_synchronization_on_update = True
            # the above line ensures that peers check for *strict* synchronization once per epoch

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            self.update_main_param_after_outer_step()
            logger.log(self.status_loglevel, f"Transitioning to epoch {self.local_epoch}")

    def _make_progress_tracker(self, target_batch_size: int, **kwargs) -> DiloCoProgressTracker:
        return DiloCoProgressTracker(
            dht=self.dht,
            prefix=self.run_id,
            target_batch_size=target_batch_size,
            client_mode=self.client_mode,
            status_loglevel=self.status_loglevel,
            start=True,
            **kwargs,
        )

    @property
    def param_groups(self) -> ParamGroups:
        """Inner optimizer is the main optimizer"""
        return self.inner_optimizer.param_groups

    def state_dict(self) -> dict:
        """we save both inner and outer optimizer states, and the local epoch"""
        state_dict_outer = self.state_averager.optimizer.state_dict()
        state_dict_outer["state"]["local_epoch"] = self.local_epoch

        state_dict_inner = self.inner_optimizer.state_dict()

        return {
            "state_dict_outer": state_dict_outer,
            "state_dict_inner": state_dict_inner,
        }

    def load_state_dict(self, state_dict: dict):
        if "local_epoch" in state_dict["state_dict_outer"]["state"]:
            self.state_averager.local_epoch = state_dict["state_dict_outer"]["state"].pop("local_epoch")

        self.state_averager.optimizer.load_state_dict(state_dict["state_dict_outer"])
        self.inner_optimizer.load_state_dict(state_dict["state_dict_inner"])

    def update_main_param_after_outer_step(self):
        """Update the main parameters with the inner optimizer step"""
        opt_parameters = [param for group in self.inner_optimizer.param_groups for param in group["params"]]
        for main_param, opt_param in zip(self.state_averager.main_parameters, opt_parameters):
            main_param.data.copy_(opt_param.data, non_blocking=True)

    def _maybe_schedule_gradient_averaging(self) -> None:
        """If next epoch is coming soon, schedule the next gradient averaging round at the estimated end of epoch"""

        if self.all_reduce_strategy == AllReduceStrategy.WAIT_FOR_ALL:
            eta_seconds = self.tracker.global_progress.eta_next_epoch - get_dht_time()
        else:
            eta_seconds = self.tracker.estimated_next_update_time

        if eta_seconds <= self.matchmaking_time:
            if (
                self.scheduled_diloco_grads is None
                or self.scheduled_diloco_grads.triggered
                or self.scheduled_diloco_grads.done()
            ):
                eta_seconds = max(eta_seconds, self.diloco_grad_averager.matchmaking_kwargs["min_matchmaking_time"])
                logger.log(self.status_loglevel, f"Pre-scheduling gradient averaging round in {eta_seconds:.2f} sec")
                self.scheduled_diloco_grads = self.diloco_grad_averager.schedule_step(timeout=self.averaging_timeout)
