from __future__ import annotations

import copy
import threading

import torch

from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.model import DTYPE_MAP, get_value_model, value_model_supports_packing
from prime_rl.utils.logger import get_logger
from prime_rl.value.batch import pack_value_inputs
from prime_rl.value.inference import predict_value_microbatches, reassemble_value_outputs
from prime_rl.value.service import ValueHTTPServer, ValueRequestService
from prime_rl.value.types import ValueEvaluationResponse
from prime_rl.value.weights import ValueWeightReceiver


class ValueEvaluatorRuntime:
    """Dedicated serving model, weight receiver, and blocking request worker."""

    def __init__(self, config: ValueFunctionConfig, evaluator_rank: int):
        if config.evaluator.placement != "dedicated":
            raise ValueError("value-evaluator can only start for dedicated evaluator placement")
        if config.model is None:
            raise ValueError("value_function.model must be resolved before starting the evaluator")
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.model = get_value_model(
            config.model,
            output_size=(config.loss.num_bins if config.loss.type == "classification" else 1),
            device=self.device,
            dtype=DTYPE_MAP[config.evaluator.dtype],
        )
        self.model.eval()
        self.model_lock = threading.Lock()
        self.update_stream = torch.cuda.Stream(device=self.device)
        self.version = -1
        self.inactive_model = None
        model_config = (
            self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        )
        self.service = ValueRequestService(
            config.evaluator,
            seq_len=config.model.seq_len,
            vocab_size=model_config.vocab_size,
        )
        self.receiver = ValueWeightReceiver(
            config.weight_broadcast,
            evaluator_rank=evaluator_rank,
            device=self.device,
        )
        self._receive_and_apply()
        if config.evaluator.double_buffer_weights:
            self.inactive_model = copy.deepcopy(self.model)
        self.update_thread = threading.Thread(target=self._update_loop, name="value-weight-updates", daemon=True)
        self.worker_thread = threading.Thread(target=self._worker_loop, name="value-request-worker", daemon=True)
        self.update_thread.start()
        self.worker_thread.start()

    def _update_loop(self) -> None:
        try:
            while True:
                self._receive_and_apply()
        except ConnectionError:
            # A finite trainer leaves a valid final serving version behind.
            get_logger().info(f"Value weight stream closed; continuing to serve version {self.version}")
        except BaseException as error:
            self.service.fail_service(error)
            get_logger().exception("Value weight receiver failed")

    def _receive_and_apply(self) -> None:
        version, layers = self.receiver.receive()
        target_model = self.inactive_model if self.inactive_model is not None else self.model

        def load_update() -> None:
            loaded: set[str] = set()
            with torch.cuda.stream(self.update_stream), torch.inference_mode():
                for layer in layers:
                    result = target_model.load_state_dict(layer, strict=False)
                    if result.unexpected_keys:
                        raise RuntimeError(f"unexpected value weights: {result.unexpected_keys}")
                    loaded.update(layer)
                expected = set(target_model.state_dict())
                if loaded != expected:
                    missing = sorted(expected - loaded)
                    extra = sorted(loaded - expected)
                    raise RuntimeError(f"incomplete value update: missing={missing[:20]}, extra={extra[:20]}")
            self.update_stream.synchronize()

        if self.inactive_model is None:
            with self.model_lock:
                load_update()
                self.version = version
                self.service.set_version(version)
        else:
            load_update()
            with self.model_lock:
                self.model, self.inactive_model = target_model, self.model
                self.version = version
                self.service.set_version(version)
        get_logger().info(f"Value evaluator adopted version {version}")

    def evaluate(self, token_ids: list[list[int]]) -> ValueEvaluationResponse:
        assert self.config.model is not None
        grid = pack_value_inputs(
            token_ids,
            seq_len=self.config.model.seq_len,
            world_size=1,
            pad_token_id=0,
            pack_sequences=value_model_supports_packing(self.model),
        )
        with self.model_lock, torch.inference_mode():
            version = self.version
            indexed = predict_value_microbatches(
                self.model,
                grid[0],
                device=self.device,
                loss=self.config.loss,
            )
        values = reassemble_value_outputs(indexed, [len(tokens) for tokens in token_ids])
        return ValueEvaluationResponse(values=values, version=version)

    def _worker_loop(self) -> None:
        while (batch := self.service.take_batch(wait_for_first=True)) is not None:
            try:
                _, update_error, _ = self.service.snapshot()
                if update_error is not None:
                    raise RuntimeError("value evaluator weight receiver failed") from update_error
                response = self.evaluate(batch.token_ids)
                self.service.complete(batch, response.values, response.version)
            except Exception as error:
                self.service.fail(batch, error)
                get_logger().exception("Value evaluation failed")


class ValueEvaluatorServer(ValueHTTPServer):
    def __init__(self, address: tuple[str, int], runtime: ValueEvaluatorRuntime):
        self.runtime = runtime
        super().__init__(address, runtime.service)
