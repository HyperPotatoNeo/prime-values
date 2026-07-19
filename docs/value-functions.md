# Value functions

Value functions are an optional, independent data plane in an RL run. They do
not add a value loss to the policy trainer. The policy trainer receives the
same token ids, sampling log probabilities, masks, and already-computed
advantages as it does without a critic.

## Quickstart

Add an empty value-function table. When a GRPO baseline is omitted, enabling
the value function resolves it to pure per-token GAE (`type = "value"`). The
empty table uses binary classification over `[0, 1]`, learning rate `1e-5`, a
one-batch FIFO replay buffer, no warmup, and independent policy/target lambdas
of `1.0`. The critic batch size inherits `orchestrator.batch_size`:

```toml
[orchestrator]
group_size = 2

[orchestrator.algo]
type = "grpo"

[value_function]
```

Run it through the normal entrypoint; the value roles are launched
automatically:

```bash
uv run rl @ examples/value_function/rl.toml
```

Use `uv run rl @ <config> --dry-run` to inspect the fully resolved value,
orchestrator, trainer, evaluator, and Slurm configs before allocating GPUs.

By default, value evaluation uses a dedicated model and GPU. To serve requests
from the live value trainer instead, add:

```toml
[value_function.evaluator]
placement = "trainer"
```

Trainer placement allocates no evaluator GPU or node. Requests queue while the
value trainer is updating, and training pauses while all trainer ranks evaluate
one queued batch. This is useful when GPU capacity matters more than independent
training and serving throughput.

## Runtime topology

With the default `evaluator.placement = "dedicated"`, enabling
`[value_function]` adds two roles:

1. After a rollout group has been finalized coherently, the orchestrator puts
   each completed trajectory on a bounded, nonblocking FIFO destined for the
   **value trainer**. The trainer pulls bounded FIFO slices as replay space is
   available, admits trajectories into a rollout-granular
   replay buffer, uniformly samples optimizer batches without replacement, and
   publishes a monotonically increasing value version after every successful
   optimizer update.
2. The dedicated **value evaluator** serves per-token values to the orchestrator. It
   adopts value-trainer weights independently of policy versions. By default,
   it loads each update into an inactive model and atomically swaps serving
   copies, so weight transfer does not hold up evaluation. A response always
   reports the single value version that produced it.

   Double buffering requires roughly twice the model parameter memory on each
   evaluator GPU. Set `evaluator.double_buffer_weights = false` to use one copy;
   in that mode, evaluation pauses while an in-place update is applied.

With `evaluator.placement = "trainer"`, the value trainer's global rank 0 also
hosts the same HTTP service. HTTP threads only enqueue requests; the distributed
trainer loop alternates complete optimizer steps with complete inference
batches. Every FSDP rank participates, and a response is produced from one
coherent value version. There is no evaluator model copy or value-weight
broadcast in this placement. Optimizer steps and checkpoints are not
preemptible, so queued requests can time out behind long trainer work.

In dedicated placement, weight updates use a CPU control notification followed
by layerwise NCCL. Evaluator replicas block on the CPU notification while idle
and enter NCCL only when a version is imminent, so the receive path never leaves
a persistent GPU communication kernel ahead of value inference.

The policy and value planes are deliberately independent:

```text
policy inference --> orchestrator ----------------------> policy trainer
                          |                                  |
                          | bounded rollout FIFO             | policy weights
                          v                                  v
                    value replay                       policy inference
                          |
                          v
                    value trainer
                     |          |
        live values  |          | value weights (dedicated)
                     v          v
               orchestrator   value evaluator --> orchestrator
```

Policy inference never waits for the value trainer. If the trainer cannot keep
up, its producer queue retains a bounded recent window and drops the oldest
pending training rollout when a newer one arrives. A completed policy
trajectory does need one evaluator result before its final advantage is known;
those requests start per rollout and overlap sibling rollout generation.
The dispatcher continues filling its bounded rollout window while this happens.
As with any bounded pipeline, an evaluator that remains slower than generation
can eventually fill that window. Dedicated evaluation never waits for value
training; trainer placement queues behind its indivisible work. Neither blocks
policy weight synchronization.

The dedicated evaluator serving copy uses BF16 parameters by default even when the
trainer keeps FP32 master weights; state loading performs the cast, while value
outputs, targets, loss reductions, and advantage math remain FP32.

## Trajectory and target contract

The orchestrator is the only component that has both environment reward and
the value prediction used for policy credit. For every trainable branch it:

1. puts the terminal environment reward on the final action token and zero on
   all earlier action tokens;
2. evaluates the causal state value for every action token;
3. computes GAE and lambda returns over action tokens only; and
4. queues the finalized rollout and its lambda returns for replay while
   stamping the selected advantage on the policy sample.

Context, prompt, and tool-response tokens are never value-loss members. Packed
sequences reset both the causal value shift and GAE recursion at every sequence
boundary. Value predictions are versioned; if an update lands while siblings
are being scored, the whole group is re-evaluated at one coherent version.

### Privileged value context

An environment task may expose an optional string field named
`value_function_prompt`. When present, the orchestrator renders that string as
a closed leading system message with the policy's canonical renderer and
prepends it to every critic branch for that rollout. The policy still samples
and trains on its original token sequence; only the value evaluator and value
trainer see the extra context. The environment therefore owns both the
privileged information and its prompt wording, while the orchestrator remains
independent of task-specific schemas such as solved grids, reference answers,
or proof sketches.

The field is static for the episode and optional per task. Omitting it leaves
the value path byte-for-byte unchanged. A non-empty field activates
conditioning without a separate prime-rl flag. Prefix tokens are masked out of
the critic loss, and evaluator outputs are projected back onto the original
policy positions before GAE and lambda returns are computed.

This is an OPSD-style rendered-token prefix, not a guarantee that every custom
chat template produces the same bytes as re-rendering one combined
conversation. Qwen system-prefix behavior and Llama BOS handling are covered;
add an integration test for another renderer family before using it in a
production experiment.

Conditioned inputs are never truncated. The rendered prefix plus the complete
policy branch must fit `value_function.model.seq_len`; otherwise the
orchestrator fails before issuing a value request or admitting the rollout to a
batch. This prevents silent loss of late action values. Configure additional
critic context length when an environment supplies substantial privileged
information. Batch logs report `value/privileged_conditioned_fraction`,
`value/privileged_prefix_tokens_mean`, and
`value/privileged_prefix_tokens_max`.

The overflow exception escapes the orchestrator's inline rollout loop. The
managed `rl` launcher then performs bounded cleanup and terminates its trainer,
value trainer, evaluator, and inference children. A standalone orchestrator
still exits nonzero, but cannot terminate services managed by another process.
Failure diagnostics report lengths and task coordinates without including the
privileged prompt content.

The policy GAE and critic target have independent lambda values:

```text
delta_t = r_t + gamma * V_(t+1) - V_t
A_policy_t = sum_l (gamma * gae_lambda)^l * delta_(t+l)
target_t = V_t + sum_l (gamma * value_target_lambda)^l * delta_(t+l)
```

Both lambdas default to `1.0`. With terminal rewards and `gamma = 1`, that
makes both streams Monte Carlo. They can be decoupled; for example,
`gae_lambda = 0.5` gives the policy a more biased, lower-variance advantage
while `value_target_lambda = 1.0` keeps the critic target Monte Carlo. The
critic target is not derived from the policy advantage after the two lambdas
diverge.

## Value head and losses

The default loss is two-bin classification over `[0, 1]`. The head emits two
logits and its scalar prediction is the softmax expectation over support
`[0, 1]`. This is equivalent to a Bernoulli prediction, but uses the same
categorical implementation as larger supports.

For a continuous target `y = 0.3`, the soft target is `[0.7, 0.3]`. If the
model predicts probabilities `[p0, p1]`, then:

```text
loss = -0.7 * log(p0) - 0.3 * log(p1)
predicted_value = 0 * p0 + 1 * p1 = p1
```

There is no rounding to class 0 or class 1: the target distribution preserves
the exact expectation `0.3`. For a support with more than two points, the same
projection puts weight on the two adjacent support points in proportion to
distance.

This is a standard categorical/discrete-regression construction, related to
the fixed-support projection in [C51](https://arxiv.org/abs/1707.06887) and the
two-hot critic targets in
[DreamerV3](https://arxiv.org/abs/2301.04104). Choosing exactly two endpoints
over `[0, 1]` as the default is specific to binary-reward RL workloads; it is
not a universal critic default. Tasks whose returns leave `[0, 1]` must widen
`reward_range`, add support bins, or select MSE regression.

MSE regression remains available with `loss.type = "mse"`. It uses one
unconstrained linear output: there is no sigmoid, clipping, or other bounding
activation.

```toml
[value_function.loss]
type = "mse"
```

Classification targets outside the configured range fail immediately rather
than being silently clipped.

## Configuration reference

| Setting | Default | Purpose |
|---|---:|---|
| `value_function.loss.type` | `classification` | `classification` for a categorical support or `mse` for scalar regression. |
| `value_function.loss.reward_range` | `[0.0, 1.0]` | Closed categorical support range. Classification targets must lie inside it. |
| `value_function.loss.num_bins` | `2` | Number of evenly spaced support atoms. |
| `value_function.gamma` | `1.0` | Discount used by both policy GAE and critic TD(lambda) targets. |
| `value_function.gae_lambda` | `1.0` | Lambda for policy advantages. |
| `value_function.value_target_lambda` | `1.0` | Independent lambda for critic targets. |
| `value_function.batch_size` | `orchestrator.batch_size` | Rollouts per critic optimizer batch. Required explicitly when the policy uses token-based batching. |
| `value_function.optim.lr` | `1e-5` | Critic AdamW learning rate. Other optimizer fields use normal trainer defaults. |
| `value_function.replay.max_updates_per_rollout` | `1` | Hard cap on the number of optimizer selections that may include one admitted rollout. FIFO eviction can retire it earlier. |
| `value_function.replay.capacity` | `max_updates_per_rollout * batch_size` | Replay capacity in rollouts. |
| `value_function.replay.refill_size` | `capacity` | Hysteretic high-water mark: training begins or resumes after the replay reaches this many rollouts. |
| `value_function.replay.seed` | `0` | Rank-0 RNG seed for uniform rollout selection. |
| `value_function.transport.max_pending_rollouts` | `2048` | Producer-side queue bound. Admission never blocks policy processing; a full queue drops its oldest pending rollout. |
| `value_function.warmup_updates` | `0` | Evaluator version required before policy batches are released. Replay behavior is unchanged during warmup. |
| `value_function.model` | policy model copy | Optional distinct critic backbone. Its tokenizer IDs must exactly match the policy tokenizer. |
| `value_function.model.seq_len` | policy `seq_len` | Critic context length; must cover the orchestrator context plus any environment-provided `value_function_prompt`. |
| `value_function.evaluator.placement` | `dedicated` | `dedicated` uses a separate serving model; `trainer` queues inference on the value-trainer GPUs. |
| `value_function.evaluator.dtype` | `bfloat16` | Dedicated serving-copy parameter dtype; inactive in trainer placement. |
| `value_function.evaluator.double_buffer_weights` | `true` | Dedicated placement only: atomically swap weight versions without pausing inference; needs roughly two model copies of GPU memory. |
| `value_function.evaluator.max_batch_tokens` | `32768` | FIFO coalescing ceiling. A larger single request is served alone. |
| `value_function.evaluator.max_pending_requests` | `64` | Maximum queued plus running requests accepted by one endpoint. |
| `value_function.evaluator.max_pending_tokens` | `1048576` | Maximum unpadded tokens across queued plus running requests. |
| `value_function.max_steps` | unset | Optional independent cap on critic updates. |
| `value_function.ckpt.interval` | unset | Optional periodic value checkpoint interval; normal completion always writes the latest version. |
| `deployment.num_value_train_gpus` / `num_value_train_nodes` | `1` | Single-node GPU count or multi-node critic-trainer node count. |
| `deployment.num_value_eval_gpus` / `num_value_eval_nodes` | placement-dependent | Defaults to one evaluator per configured replica in dedicated placement and zero in trainer placement. |

Existing configs may migrate incrementally. The deprecated
`value_function.updates_per_batch` key is translated to
`value_function.replay.max_updates_per_rollout`, with the new per-rollout cap
semantics; setting both to different values fails validation. The deprecated
`value_function.transport.type = "zmq_latest"` tag is translated to `"zmq"`.
Both translations emit `FutureWarning` and will be removed in a future release.

Every field is also available as a dotted CLI override, for example:

```bash
uv run rl @ rl.toml \
  --value-function.gae-lambda 0.5 \
  --value-function.value-target-lambda 1.0 \
  --value-function.optim.lr 1e-5
```

## Baselines

Without `[value_function]`, GRPO keeps the existing group-mean default. With a
value function enabled, an omitted GRPO baseline resolves to `value`. Any
explicit baseline remains unchanged. The `baseline` is a discriminated
configuration with non-value and value-backed choices:

- `mean`: standard GRPO, `A = R - mean(R)`;
- `leave_one_out`: `A_i = R_i - mean(R_{j != i})`;
- `value`: pure per-token GAE.

Length penalties remain group-credit-only; when `[value_function]` is enabled,
set `baseline.type` explicitly to `mean` or `leave_one_out` before configuring
one.

`value` is invalid unless `[value_function]` is enabled. Explicitly selecting
`mean` or `leave_one_out` with a value function is valid and trains the critic
for diagnostics or a later baseline change.

## Staleness and overload

Value staleness is independent of policy off-policy level. Each evaluator
response records one `value_version`; every replayed rollout retains its source
policy and evaluator versions, and an optimizer batch reports their ranges.
Evaluator coherence is enforced within every rollout group used for policy
credit, while uniformly mixing independently labeled groups in a critic update
is allowed. Lambda-return targets are frozen when the rollout is finalized, so
the source-value lag is important when `value_target_lambda < 1`.

The producer queue and replay buffer have deliberately different jobs. The
producer queue prevents critic throughput from backpressuring inference. It is
bounded by `transport.max_pending_rollouts`, keeps rollouts droppable until the
trainer requests a bounded admission slice, and drops the oldest still-pending
rollout on overflow. One trainer-credited response of at most one optimizer
batch may be in flight outside the queue; unsolicited rollouts are never moved
into transport buffers. Producer publication still performs finite local
MessagePack encoding on the orchestrator event loop; nonblocking means it never
waits for trainer or network progress.
The value evaluator request service is not part of this queue: policy
advantages retain their existing completion, timeout, and overload semantics.

The replay buffer bounds both age and reuse. Admission and capacity eviction
are FIFO, but each optimizer batch is a uniform sample of distinct resident
rollouts; sampling does not refresh FIFO age. A rollout is retired after
`max_updates_per_rollout` selections or when newer admission evicts it,
whichever happens first. The cap is therefore not a promise that every rollout
will receive that many updates. Under sustained producer pressure, each trainer
turn admits up to one optimizer batch before sampling, favoring recent data and
often realizing fewer than the maximum selections. When arrivals are slower,
resident rollouts can instead be reused up to the cap.

Replay readiness is hysteretic. In the filling state, the trainer waits until
`refill_size` resident rollouts are available. It then keeps sampling while at
least one full optimizer batch remains. Once occupancy falls below
`batch_size`, it returns to filling and waits for `refill_size` again. With the
defaults and `max_updates_per_rollout = 1`, capacity and refill size both equal
one optimizer batch, so the selected cohort and insertion order match the
ordinary one-update cadence when the producer is not overloaded.

## Monitoring

Enabling the top-level `[wandb]` block propagates the same W&B project, run,
group, tags, and shared-run identity to the value trainer. Value metrics are
logged after every critic optimizer update; evaluator and value-queue metrics
are also logged by the orchestrator.

| Metric | Interpretation |
|---|---|
| `value/loss`, `value/mae`, `value/mse`, `value/rmse` | Training objective and prediction-space error views. For classification, `value/loss` is cross entropy while the error metrics use the softmax expectation. |
| `value/bias`, `value/explained_variance` | Signed prediction error and fraction of target variance explained. Explained variance is reported as zero for a constant-target batch. |
| `value/prediction_{mean,std,min,max}` | Critic predictions on the optimizer batch. |
| `value/target_{mean,std,min,max}` | TD(lambda) target distribution. |
| `value/accuracy`, `value/entropy`, `value/confidence` | Classification diagnostics. Accuracy compares the most probable support atom to the target's nearest atom; error metrics remain more informative for continuous soft targets. |
| `value/version`, `value/source_value_version_{min,max,spread}`, `value/source_value_lag_{min,max}` | Trainer version and the evaluator-version provenance range represented in the optimizer batch. The unsuffixed source-version metric is the maximum and the unsuffixed lag is the maximum. |
| `value/source_policy_version_{min,max,spread}`, `value/source_rollout_id_{min,max,spread}` | Policy-version and producer-order provenance represented in the sampled optimizer batch. |
| `value/replay_attempt_{min,mean,max}` | Selection counts after reserving the sampled rollouts. |
| `value/replay_size_{rollouts,samples,tokens}`, `value/replay_{capacity,refill_size}_rollouts`, `value/replay_ready` | Replay occupancy, configured hysteresis, and readiness after the selection. Samples count branches; tokens measure stored sequence payload. |
| `value/replay_{admitted,attempts,retired,evicted}_total` | Cumulative replay admission, selection, max-cap retirement, and FIFO-capacity eviction counts. |
| `value/batch_{tokens,rollouts,samples}`, `value/total_{tokens,samples}` | Per-update and cumulative critic data volume. Samples count branches; rollouts count episodes used to trigger the batch. |
| `value/update_seconds`, `value/tokens_per_second` | Value optimizer performance. |
| `value/service_pending_{requests,tokens}`, `value/service_{oldest,max}_wait_seconds` | Trainer-placement queue pressure, including selected work until completion. |
| `value/service_{admitted,rejected_full,expired,abandoned,completed,failed}` | Trainer-placement request outcomes and overload behavior. |
| `value/checkpoint_seconds`, `value/service_{inference,training,idle}_seconds`, `value/service_inference_{batches,tokens}` | Trainer-placement checkpoint/scheduling time and inference volume. |
| `optim/lr`, `optim/grad_norm`, `optim/zero_grad_ratio` | Critic optimizer health. |
| `value/evaluator_{requests,sequences,tokens,errors,error_rate}` | Cumulative evaluator service volume and failures. |
| `value/evaluator_latency_seconds_{mean,max}` | End-to-end HTTP evaluation latency, including dynamic-batcher waiting. |
| `value/evaluator_version`, `value/evaluator_version_spread` | Evaluator versions represented in a policy batch. A nonzero spread is corrected by coherent group re-evaluation before advantages are stamped. |
| `value/privileged_conditioned_fraction`, `value/privileged_prefix_tokens_{mean,max}` | Fraction of value-backed policy rollouts carrying environment-provided privileged context and the number of tokens inserted into each conditioned rollout. |
| `value/rollout_{prediction,advantage,target}_{mean,std,min,max}` | Values used on the actual policy rollouts, before the critic optimizer update. |
| `value/rollout_queue_{enqueued,sent,dropped_oldest,pending,capacity}`, `value/rollout_queue_drop_rate` | Producer-side rollout flow and bounded-queue pressure. `sent` means the credited response was accepted by ZeroMQ, not acknowledged as replay admission; a nonzero drop rate means critic training lost old pending rollouts, not that policy inference stopped. |
| `value/rollout_queue_pending_{bytes,tokens}`, `value/rollout_responder_failures` | Encoded host-memory pressure and pull-responder health. The local FIFO remains within its configured rollout capacity while the trainer is not admitting data. |

## Warmup

`warmup_updates` remains a value-version barrier, not a fixed number of rollout
batches. Replay admission, refill, sampling, and per-rollout update caps are the
same before and after the barrier. The dispatcher and policy inference keep
generating and publishing critic rollouts while policy batches are withheld;
normal policy shipping begins as soon as the evaluator adopts the required
version. With a replay larger than one batch, initial warmup therefore waits
for the configured refill threshold. Loading a value checkpoint may satisfy
some or all of the version barrier, but replay still starts cold.

## Initialization and checkpoints

By default the value transformer is initialized from the policy base model and
uses a newly initialized scalar or categorical value head. `model.name` can
select a different compatible base model. Because the evaluator consumes policy
token IDs directly, a different model must use the identical tokenizer
vocabulary and ID mapping. The launcher compares both vocabularies and special
token IDs before starting; `value_function.tokenizer_name` can name the
tokenizer a critic expects when it is not stored with the critic model. Its
`seq_len` must cover the orchestrator sequence length. A value checkpoint
restores the value model, optimizer, scheduler, and independent value version.

Policy and value checkpoints have separate directories and progress counters.
Resuming one never rewrites the other's step. Dedicated evaluator weights are a
serving copy reconstructed from the value checkpoint plus subsequent trainer
publications; trainer placement serves directly from the restored live model.
Replay contents, sampling RNG state, and the producer queue are not
checkpointed. A resumed value trainer restores model, optimizer, scheduler, and
version, then refills a new replay from fresh finalized rollouts.
On normal policy completion, the value trainer finishes its selected operation
and writes a final value checkpoint before the launcher tears down the value
plane.

## Multi-node layout

The critic trainer uses the same FSDP/`torchrun` substrate as the policy trainer
and can span multiple dedicated nodes. Evaluators can be replicated on separate
nodes:

```toml
[deployment]
type = "multi_node"
gpus_per_node = 8
num_infer_nodes = 1
num_train_nodes = 1
num_value_train_nodes = 2
num_value_eval_nodes = 2

[value_function.weight_broadcast]
evaluator_world_size = 2

[value_function.evaluator]
base_url = ["http://value-eval-0:29612", "http://value-eval-1:29612"]
```

The launcher replaces evaluator hosts with the allocated nodes. The critic
world size must be divisible by its `model.dp_replicate`; this is validated
before submission. The value trainer is sharded across all GPUs on its nodes.
Each evaluator replica is currently an unsharded, single-GPU model and must fit
on one GPU (twice with double buffering); the launcher starts one replica per
evaluator node, so remaining GPUs on that dedicated node are presently unused.

Trainer placement needs no evaluator nodes:

```toml
[value_function.evaluator]
placement = "trainer"

[deployment]
num_value_eval_nodes = 0 # optional; this is the resolved default
```

The launcher routes the orchestrator to the value-trainer master and omits the
evaluator process and weight broadcast. The value trainer stays alive after its
own `max_steps` in serve-only mode until the policy run finishes.

## Scope

Value functions support text models and one RL run. LoRA, VLM inputs,
expert-router replay, and multi-run value sharing fail validation rather than
silently behaving incorrectly. Trainer placement supports dense FSDP
with one data-parallel replica and no context/expert parallelism, FSDP parameter
CPU offload, DeepEP, or FP8. Dedicated placement retains the broader existing
value-model topology. The transport, evaluator client, baseline, and value-model
APIs remain separate from the policy trainer contract.

A distributed-rank failure or independent value-transport peer process
exit/restart is fatal rather than hot-recoverable; the managed launcher tears
down all child processes. In particular, restarting a rollout producer while
the trainer is awaiting its credited response does not repair that outstanding
REQ state. Rank 0 marks the HTTP service failed when it observes a distributed
error; a request already executing on a failed peer can remain pending until
the process-group or request timeout.
