from __future__ import annotations

import copy
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec
import torch

from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.trainer.model import DTYPE_MAP, get_value_model, predict_value, value_model_supports_packing
from prime_rl.utils.logger import get_logger
from prime_rl.value.batch import pack_value_samples
from prime_rl.value.math import align_value_logits, predict_values
from prime_rl.value.types import (
    ValueEvaluationRequest,
    ValueEvaluationResponse,
    ValueTrainingSample,
    ValueVersionResponse,
)
from prime_rl.value.weights import ValueWeightReceiver


class ValueEvaluatorRuntime:
    def __init__(self, config: ValueFunctionConfig, evaluator_rank: int):
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
        self.update_error: BaseException | None = None
        self.inactive_model = None
        self.receiver = ValueWeightReceiver(
            config.weight_broadcast,
            evaluator_rank=evaluator_rank,
            device=self.device,
        )
        self._receive_and_apply()
        if config.evaluator.double_buffer_weights:
            self.inactive_model = copy.deepcopy(self.model)
        self.update_thread = threading.Thread(target=self._update_loop, name="value-weight-updates", daemon=True)
        self.update_thread.start()

    def _update_loop(self) -> None:
        try:
            while True:
                self._receive_and_apply()
        except ConnectionError:
            # A finite value trainer, or a trainer that exits cleanly with the
            # policy run, leaves a perfectly usable final serving version.
            get_logger().info(f"Value weight stream closed; continuing to serve version {self.version}")
        except BaseException as error:
            self.update_error = error
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
            # Memory-saving mode updates the live model and must exclude
            # inference for the complete transfer.
            with self.model_lock:
                load_update()
                self.version = version
        else:
            # Double-buffer mode receives on a dedicated stream while the
            # active copy serves; only the pointer/version swap is serialized.
            load_update()
            with self.model_lock:
                self.model, self.inactive_model = target_model, self.model
                self.version = version
        get_logger().info(f"Value evaluator adopted version {version}")

    def evaluate(self, token_ids: list[list[int]]) -> ValueEvaluationResponse:
        if self.update_error is not None:
            raise RuntimeError("value evaluator weight receiver failed") from self.update_error
        assert self.config.model is not None
        samples = [ValueTrainingSample(tokens, [True] * len(tokens), [0.0] * len(tokens)) for tokens in token_ids]
        grid = pack_value_samples(
            samples,
            seq_len=self.config.model.seq_len,
            world_size=1,
            pad_token_id=0,
            pack_sequences=value_model_supports_packing(self.model),
        )
        results: list[list[float] | None] = [None] * len(samples)
        with self.model_lock, torch.inference_mode():
            version = self.version
            for micro_batch in grid[0]:
                input_ids = torch.tensor(micro_batch.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                position_ids = torch.tensor(micro_batch.position_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                logits = predict_value(self.model, input_ids, position_ids)
                logits = align_value_logits(logits, micro_batch.sequence_lengths)
                values = predict_values(logits, self.config.loss).reshape(-1)
                offset = 0
                for sample_index, length in zip(micro_batch.sample_indices, micro_batch.sequence_lengths, strict=True):
                    if sample_index >= 0:
                        results[sample_index] = values[offset : offset + length].float().cpu().tolist()
                    offset += length
        if any(result is None for result in results):
            raise RuntimeError("value evaluator failed to materialize every requested sequence")
        return ValueEvaluationResponse(values=results, version=version)  # type: ignore[arg-type]


@dataclass
class _WorkItem:
    request: ValueEvaluationRequest
    future: Future[ValueEvaluationResponse]


class DynamicValueBatcher:
    def __init__(self, runtime: ValueEvaluatorRuntime):
        self.runtime = runtime
        self.config = runtime.config.evaluator
        self.queue: queue.Queue[_WorkItem] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="value-dynamic-batcher", daemon=True)
        self.thread.start()

    def submit(self, request: ValueEvaluationRequest) -> ValueEvaluationResponse:
        future: Future[ValueEvaluationResponse] = Future()
        self.queue.put(_WorkItem(request, future))
        return future.result(timeout=self.config.request_timeout)

    def _loop(self) -> None:
        while True:
            first = self.queue.get()
            items = [first]
            tokens = sum(map(len, first.request.token_ids))
            deadline = time.monotonic() + self.config.batch_wait_ms / 1000.0
            while tokens < self.config.max_batch_tokens:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self.queue.get(timeout=timeout)
                except queue.Empty:
                    break
                item_tokens = sum(map(len, item.request.token_ids))
                if tokens + item_tokens > self.config.max_batch_tokens:
                    self.queue.put(item)
                    break
                items.append(item)
                tokens += item_tokens

            counts = [len(item.request.token_ids) for item in items]
            try:
                response = self.runtime.evaluate([tokens for item in items for tokens in item.request.token_ids])
                offset = 0
                for item, count in zip(items, counts, strict=True):
                    item.future.set_result(
                        ValueEvaluationResponse(
                            values=response.values[offset : offset + count],
                            version=response.version,
                        )
                    )
                    offset += count
            except BaseException as error:
                for item in items:
                    item.future.set_exception(error)


class ValueEvaluatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: ValueEvaluatorRuntime):
        self.runtime = runtime
        self.batcher = DynamicValueBatcher(runtime)
        super().__init__(address, ValueEvaluatorHandler)


class ValueEvaluatorHandler(BaseHTTPRequestHandler):
    request_decoder = msgspec.msgpack.Decoder(type=ValueEvaluationRequest)
    encoder = msgspec.msgpack.Encoder()

    @property
    def value_server(self) -> ValueEvaluatorServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args) -> None:
        get_logger().debug(format % args)

    def do_GET(self) -> None:
        if self.path == "/health":
            if self.value_server.runtime.update_error is not None:
                self._write(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b"value weight receiver failed",
                    content_type="text/plain",
                )
                return
            self._write(HTTPStatus.OK, b"ok", content_type="text/plain")
            return
        if self.path == "/version":
            self._write(
                HTTPStatus.OK,
                self.encoder.encode(ValueVersionResponse(version=self.value_server.runtime.version)),
            )
            return
        self._write(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")

    def do_POST(self) -> None:
        if self.path != "/evaluate":
            self._write(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            request = self.request_decoder.decode(self.rfile.read(length))
            response = self.value_server.batcher.submit(request)
            self._write(HTTPStatus.OK, self.encoder.encode(response))
        except Exception as error:
            get_logger().exception("Value evaluation failed")
            self._write(HTTPStatus.INTERNAL_SERVER_ERROR, str(error).encode(), content_type="text/plain")

    def _write(self, status: HTTPStatus, payload: bytes, *, content_type: str = "application/msgpack") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
