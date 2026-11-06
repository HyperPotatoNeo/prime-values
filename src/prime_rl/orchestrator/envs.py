"""Client-owned native tasksets with server-owned agent execution.

Workers reconstruct each selected task from its serialized data and return a
completed canonical Episode through the single-trace adapter; task
data remains opaque to policy and value consumers.
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
import os
import queue
import sys
from collections.abc import Iterator, Sequence
from itertools import islice
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import verifiers.v1 as vf
from verifiers.v1.serve import EnvClient

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.algo import Algorithm, build_algorithm
from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.sampler import Sampler
from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.value import ValueFunctionConfig

# Legacy workers may download their dataset before reporting an address.
ENV_SERVER_SPAWN_TIMEOUT = 600.0


def _run_env_server(
    *,
    log_file: str,
    log_level: str,
    json_logging: bool,
    **kwargs,
) -> None:
    """Spawned-process entry point: redirect this process's output to ``log_file`` (the
    server's logging + any subprocess-runtime output), then serve via ``serve_env``. The
    worker-pool sizing arrives in ``kwargs`` (``max_workers`` / ``multiplex`` / ``elastic``
    from the env's ``pool``). ``serve_env`` applies ``log_setup`` here and in every spawned
    worker; a worker inherits this process's redirected stdout/stderr, so its per-rollout
    logs reach ``log_file`` too. Top-level so it stays picklable for the ``spawn`` start
    method."""
    from functools import partial

    from verifiers.v1.serve import serve_env

    from prime_rl.orchestrator.utils import setup_env_server_logging

    fh = open(log_file, "w", buffering=1)
    os.dup2(fh.fileno(), sys.stdout.fileno())
    os.dup2(fh.fileno(), sys.stderr.fileno())
    serve_env(
        log_setup=partial(setup_env_server_logging, log_level, json_logging),
        **kwargs,
    )


class Env:
    """Owns native task selection; workers own agent/harness execution."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}
        self.num_tasks: int | None = 0
        self.tasks: Iterator[vf.Task] | None = None
        self._task_class: type[vf.Task] | None = None
        self.requires_group_scoring: bool = False
        self._env_client: EnvClient | None = None
        self._env_server_process: BaseProcess | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env_client(self) -> EnvClient:
        if self._env_client is None:
            raise RuntimeError(f"Env {self.name} not started — call start() first.")
        return self._env_client

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        """Load native tasks once, then connect to their execution worker."""
        taskset = vf.load_taskset(self.config.env.taskset)
        self._task_class = type(taskset).task_type()
        if self.config.address is not None:
            from verifiers.v1.tasksets.harbor import HarborTask

            if issubclass(self._task_class, HarborTask):
                raise ValueError(
                    f"Env {self.name}: external Harbor workers require verified shared grading-artifact paths; "
                    "this launcher supports spawned local Harbor workers only"
                )
        if taskset.transform is not None:
            raise ValueError(
                "Taskset views are not supported by this sampler; configure dataset selection in the taskset"
            )

        def selected_tasks():
            # Preserve caller-supplied identities; new native loaders may leave idx unset.
            # This deliberately does not use Taskset.__iter__, which overwrites every idx.
            for position, task in enumerate(taskset.load()):
                update = {} if taskset.system_prompt is None else {"system_prompt": taskset.system_prompt}
                if task.data.idx is None:
                    update["idx"] = position
                yield self.validate_task(task.with_data(**update) if update else task)

        if type(taskset).INFINITE:
            if isinstance(self.config, EvalEnvConfig) and self.config.num_examples < 0:
                raise ValueError(f"Eval env {self.name} has an infinite taskset; set num_examples to bound it")
            self.tasks = selected_tasks()
            self.num_tasks = None
        else:
            tasks = await asyncio.to_thread(lambda: list(selected_tasks()))
            self.tasks = iter(tasks)
            self.num_tasks = len(tasks)
        external = self.config.address is not None
        address = self.config.address or await self._spawn(log_dir, log_level or "INFO", json_logging)
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = EnvClient(address=address)
        # Spawned workers report after binding. External workers need a health handshake.
        if external:
            await self.env_client.wait_for_server_startup()
        get_logger().info(
            f"Env {self.name} ready: num_tasks={self.num_tasks} group_scoring={self.requires_group_scoring}"
        )

    def validate_task(self, task: vf.Task) -> vf.Task:
        """Reject task behavior or data that the stateless worker cannot reconstruct."""
        if self._task_class is None:
            raise RuntimeError(f"Native task type for {self.name} has not been loaded")
        if type(task) is not self._task_class:
            raise ValueError(
                f"Env {self.name} yielded {type(task).__name__}; worker reconstructs {self._task_class.__name__}"
            )
        if task.config != self.config.env.taskset.task:
            raise ValueError(f"Env {self.name} yielded per-row task config that differs from its worker config")
        if type(task.data.idx) is not int:
            raise ValueError(f"Env {self.name} requires an integer task.data.idx for rollout records")
        data_class = self._task_class.data_type()
        if any(field.exclude for field in data_class.model_fields.values()):
            raise ValueError(f"Env {self.name} task data contains excluded fields that cannot survive dispatch")
        payload = task.data.model_dump(mode="json")
        rebuilt = data_class.model_validate(payload)
        if rebuilt != task.data or rebuilt.model_dump(mode="json") != payload:
            raise ValueError(f"Env {self.name} task data changes when reconstructed by its worker")
        from verifiers.v1.tasksets.harbor import HarborTask

        if isinstance(task, HarborTask) and not (Path(task.data.task_dir) / "tests").is_dir():
            raise ValueError(f"Env {self.name}: Harbor grading artifacts are missing at {task.data.task_dir!r}/tests")
        return task

    async def _spawn(self, log_dir: Path, log_level: str, json_logging: bool) -> str:
        """Spawn a worker for agent execution (and legacy dataset ownership).
        The server binds an OS-assigned port (``:0``) and reports the concrete
        address back over a queue — no free-port guess, no TOCTOU race. Its output
        goes to ``<log_dir>/<name>.log`` (``log_dir`` is already the train/eval-split
        ``.../logs/envs/{train,eval}`` the orchestrator passes in)."""
        ctx = mp.get_context("spawn")
        address_queue: mp.Queue = ctx.Queue()
        log_file = log_dir / f"{self.name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        get_logger().debug(f"Spawning env server {self.name} (id={self.config.env_id}, log={log_file})")
        server_kwargs = dict(
            config_data=self.config.env.model_dump(mode="json"), max_concurrent=self.config.max_concurrent
        )
        process = ctx.Process(
            target=_run_env_server,
            kwargs=dict(
                log_file=str(log_file),
                log_level=log_level,
                json_logging=json_logging,
                **vf.pool_serve_kwargs(self.config.pool),
                address="tcp://127.0.0.1:0",
                address_queue=address_queue,
                **server_kwargs,
            ),
            daemon=False,
        )
        process.start()
        self._env_server_process = process
        try:
            address = await asyncio.to_thread(address_queue.get, timeout=ENV_SERVER_SPAWN_TIMEOUT)
        except queue.Empty:
            raise RuntimeError(f"Env server {self.name} did not report its address within {ENV_SERVER_SPAWN_TIMEOUT}s")
        finally:
            address_queue.close()
            address_queue.join_thread()
        get_logger().debug(f"Env server {self.name} bound at {address}")
        return address

    def _sampling(self, cache_salt: str | None) -> vf.SamplingConfig:
        sampling = {**self.sampling_args}
        if cache_salt is not None:
            sampling["extra_body"] = {
                **sampling.get("extra_body", {}),
                "cache_salt": cache_salt,
            }
        return vf.SamplingConfig(**sampling)

    async def run_episode(
        self,
        client: vf.ClientConfig,
        task_idx: int,
        model_name: str,
        cache_salt: str | None,
        task_data: dict | None = None,
    ) -> vf.WireEpisode:
        """Await the upstream client's complete assembled episode, never partial deltas."""
        if task_data is None:
            raise ValueError("Native requests require the selected owned task_data")
        return await self.env_client.run(
            task_data=task_data,
            client=client,
            model=model_name,
            sampling=self._sampling(cache_salt),
        )

    async def run_rollout(
        self,
        client: vf.ClientConfig,
        task_idx: int,
        model_name: str,
        cache_salt: str | None,
        task_data: dict | None = None,
    ) -> Rollout:
        """Project a canonical single-trace episode onto the rollout consumer boundary."""
        return from_episode(
            await self.run_episode(client, task_idx, model_name, cache_salt, task_data=task_data),
            task_idx=task_idx,
            training_projection=self.config.training_projection if isinstance(self.config, TrainEnvConfig) else None,
        )

    async def run_group(
        self,
        client: vf.ClientConfig,
        task_idx: int,
        model_name: str,
        group_size: int,
        cache_salt: str | None,
    ) -> list[Rollout]:
        raise ValueError("Joint legacy group scoring requires the frozen current-stack producer")

    def shutdown(self) -> None:
        if self._env_server_process is None:
            return
        self._env_server_process.terminate()
        self._env_server_process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig, sampler: Sampler, algorithm: Algorithm):
        super().__init__(config)
        self.sampler = sampler
        self.algorithm = algorithm
        self.sampling_args = sampler.sampling_args(config.sampling.to_sampling_args())


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()
        self.examples: list[dict] = []

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        await super().start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        n = self.config.num_examples
        if self.tasks is None:
            count = self.num_tasks if n < 0 else min(n, self.num_tasks)
            self.examples = [{"task_idx": i} for i in range(count)]
            return
        if self.num_tasks is None and n < 0:
            raise ValueError(f"Eval env {self.name} has an infinite taskset; set num_examples to bound it")
        tasks = list(self.tasks) if n < 0 else list(islice(self.tasks, n))
        self.examples = [{"task_idx": task.data.idx, "task": task} for task in tasks]


EnvT = TypeVar("EnvT", bound=Env)


class Envs(Generic[EnvT]):
    """Base container for a set of Env instances."""

    _envs: dict[str, EnvT]

    @property
    def names(self) -> list[str]:
        return list(self._envs.keys())

    @property
    def configs(self) -> list[EnvConfig]:
        return [env.config for env in self._envs.values()]

    def get(self, name: str) -> EnvT:
        return self._envs[name]

    def __iter__(self) -> Iterator[EnvT]:
        return iter(self._envs.values())

    def __len__(self) -> int:
        return len(self._envs)

    async def start(self, log_dir: Path, log_level: str | None = None, json_logging: bool = False) -> None:
        """Spawn env servers (where needed) and connect, one at a time. Each server
        binds an OS-assigned port and reports it back, so there's no port race."""
        for env in self:
            await env.start(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Terminate all spawned env server processes."""
        processes = [env._env_server_process for env in self if env._env_server_process is not None]
        if not processes:
            return
        logger = get_logger()
        logger.debug(f"Shutting down {len(processes)} env server(s)")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=25)
            if p.is_alive():
                logger.warning(f"Env server {p.pid} did not exit after 25s, force killing")
                p.kill()
                p.join(timeout=5)
        for env in self:
            env._env_server_process = None


class TrainEnvs(Envs[TrainEnv]):
    """Collection of training environments, each paired with its rollout
    :class:`Sampler` and runtime :class:`Algorithm`, built from the env's
    resolved algorithm config."""

    def __init__(
        self,
        configs: Sequence[TrainEnvConfig],
        *,
        policy_pool,
        renderer_config=None,
        value_evaluator=None,
        value_config=None,
        policy_seq_len=None,
    ):
        self._envs: dict[str, TrainEnv] = {}
        for config in configs:
            assert config.algo is not None, "TrainEnvConfig.algo must be resolved before env construction"
            env = TrainEnv(
                config,
                Sampler(config.algo.sampling, policy_pool, renderer_config),
                build_algorithm(
                    config.algo,
                    policy_pool,
                    value_evaluator=value_evaluator,
                    value_config=value_config,
                    policy_seq_len=policy_seq_len,
                ),
            )
            self._envs[env.name] = env

    def validate_group_value_context(self, value_config: ValueFunctionConfig | None) -> None:
        """Reject runtime reward semantics that can reveal the omitted rollout."""
        if value_config is None or value_config.privileged_context != "group_leave_one_out":
            return
        group_scored = [
            env.name for env in self if env.algorithm.value_evaluator is not None and env.requires_group_scoring
        ]
        if group_scored:
            raise ValueError(
                "group_leave_one_out value context does not support environments with group-dependent rewards: "
                + ", ".join(group_scored)
            )


class EvalEnvs(Envs[EvalEnv]):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, EvalEnv] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
