# Value functions

Value functions are an optional, independent data plane in an RL run. They do
not add a value loss to the policy trainer. The policy trainer receives the
same token ids, sampling log probabilities, masks, and already-computed
advantages as it does without a critic.

## Quickstart

Add an empty value-function table and select a value-backed GRPO baseline. The
empty table uses binary classification over `[0, 1]`, learning rate `1e-5`, one
critic update per full rollout batch, and independent policy/target lambdas of
`1.0`. The critic batch size inherits `orchestrator.batch_size`:

```toml
[orchestrator]
group_size = 2

[orchestrator.algo]
type = "grpo"

[orchestrator.algo.baseline]
type = "linear_mix" # "value" and "tether" are also value-backed
# group = "leave_one_out" is the linear_mix/tether default

[value_function]
```

Run it through the normal entrypoint; the value roles are launched
automatically:

```bash
uv run rl @ examples/value_function/rl.toml
```

Use `uv run rl @ <config> --dry-run` to inspect the fully resolved value,
orchestrator, trainer, evaluator, and Slurm configs before allocating GPUs.

## Runtime topology

Enabling `[value_function]` adds two roles:

1. The orchestrator accumulates completed trajectories into full critic
   rollout batches before filtering. The **value trainer** receives those
   batches on a bounded, latest-only queue, runs `updates_per_batch` optimizer
   updates on the newest available batch, discards it, and publishes a
   monotonically increasing value version after every optimizer update.
2. The **value evaluator** serves per-token values to the orchestrator. It
   adopts value-trainer weights independently of policy versions. By default,
   it loads each update into an inactive model and atomically swaps serving
   copies, so weight transfer does not hold up evaluation. A response always
   reports the single value version that produced it.

   Double buffering requires roughly twice the model parameter memory on each
   evaluator GPU. Set `evaluator.double_buffer_weights = false` to use one copy;
   in that mode, evaluation pauses while an in-place update is applied.

Weight updates use a CPU control notification followed by layerwise NCCL.
Evaluator replicas block on the CPU notification while idle and enter NCCL
only when a version is imminent, so the receive path never leaves a persistent
GPU communication kernel ahead of value inference.

The policy and value planes are deliberately independent:

```text
policy inference --> orchestrator ----------------------> policy trainer
                          |                                  |
                          | latest-only value batches        | policy weights
                          v                                  v
                    value trainer --> value weights    policy inference
                          |
                          v
                    value evaluator --> values --> orchestrator advantages
```

Policy inference never waits for the value trainer. If the trainer cannot keep
up, intermediate value batches are replaced by newer batches. A completed
policy trajectory does need one evaluator result before its final advantage is
known; those requests start per rollout and overlap sibling rollout generation.
The dispatcher continues filling its bounded rollout window while this happens.
As with any bounded pipeline, an evaluator that remains slower than generation
can eventually fill that window; it never waits for value-training updates or
blocks policy weight synchronization.

The evaluator serving copy uses BF16 parameters by default even when the
trainer keeps FP32 master weights; state loading performs the cast, while value
outputs, targets, loss reductions, and advantage math remain FP32.

## Trajectory and target contract

The orchestrator is the only component that has both environment reward and
the value prediction used for policy credit. For every trainable branch it:

1. puts the terminal environment reward on the final action token and zero on
   all earlier action tokens;
2. evaluates the causal state value for every action token;
3. computes GAE and lambda returns over action tokens only; and
4. queues lambda returns into a full critic rollout batch while stamping the
   selected advantage on the policy sample.

Context, prompt, and tool-response tokens are never value-loss members. Packed
sequences reset both the causal value shift and GAE recursion at every sequence
boundary. Value predictions are versioned; if an update lands while siblings
are being scored, the whole group is re-evaluated at one coherent version.

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
| `value_function.updates_per_batch` | `1` | Optimizer updates made on one recent full rollout batch before discarding it. |
| `value_function.warmup_updates` | `0` | Evaluator version required before policy batches are released. Generation continues during warmup. |
| `value_function.model` | policy model copy | Optional distinct critic backbone. Its tokenizer IDs must exactly match the policy tokenizer. |
| `value_function.model.seq_len` | policy `seq_len` | Critic context length; must cover the orchestrator context. |
| `value_function.evaluator.dtype` | `bfloat16` | Evaluator serving-copy parameter dtype; outputs and advantage math stay FP32. |
| `value_function.evaluator.double_buffer_weights` | `true` | Atomically swap weight versions without pausing inference; needs roughly two model copies of GPU memory. |
| `value_function.evaluator.max_batch_tokens` | `32768` | Dynamic evaluator batch token ceiling. |
| `value_function.max_steps` | unset | Optional independent cap on critic updates. |
| `value_function.ckpt.interval` | unset | Optional periodic value checkpoint interval; normal completion always writes the latest version. |
| `deployment.num_value_train_gpus` / `num_value_train_nodes` | `1` | Single-node GPU count or multi-node critic-trainer node count. |
| `deployment.num_value_eval_gpus` / `num_value_eval_nodes` | `1` | Evaluator replica capacity; multi-node currently launches one single-GPU replica per node. |

Every field is also available as a dotted CLI override, for example:

```bash
uv run rl @ rl.toml \
  --value-function.gae-lambda 0.5 \
  --value-function.value-target-lambda 1.0 \
  --value-function.optim.lr 1e-5
```

## Baselines

GRPO keeps the existing group-mean baseline by default. Its `baseline` is a
discriminated configuration with non-value and value-backed choices:

- `mean`: standard GRPO, `A = R - mean(R)`;
- `leave_one_out`: `A_i = R_i - mean(R_{j != i})`;
- `value`: pure per-token GAE;
- `linear_mix`: an affine blend of a group advantage and GAE using one static
  coefficient, `A = (1 - rho) * A_group + rho * A_value`;
- `tether`: a clipped two-factor correction anchored on a group baseline.

Both `linear_mix` and `tether` accept `group = "mean"` or
`group = "leave_one_out"`; **leave-one-out is the default for both**.

TETHER forms the complete baseline and clips once at the end:

```text
b_t = clip(B_group + alpha * (V_0 - B_group) + rho * (V_t - V_0), low, high)
A_t = R - b_t
```

The anchor correction and progress term are not clipped separately.
Without an `adaptive` table, `alpha` and `rho` are static finite coefficients.
They are not restricted to `[0, 1]` because calibrated control-variate
coefficients may legitimately lie outside that interval (including being
negative).

### Adaptive TETHER coefficients

An empty nested table enables the online two-factor fit:

```toml
[orchestrator.algo.baseline]
type = "tether"
group = "leave_one_out"
reward_range = [0.0, 1.0]

[orchestrator.algo.baseline.adaptive]
# batch_size = value_function.batch_size
# ridge = 1e-6
# ema_decay = 0.9
# initial_alpha = 0.0
# initial_rho = 0.0
```

Adaptive mode ignores the static `baseline.alpha` and `baseline.rho` fields;
use `adaptive.initial_alpha` and `adaptive.initial_rho` when a nonzero start is
intentional. Their defaults are zero. With the required leave-one-out anchor,
zero/zero is exactly the LOO sibling-mean baseline, not the own-inclusive GRPO
group mean.

For every trainable token, the estimator fits the no-intercept ridge problem

```text
y       = R - B_LOO
x_alpha = V_0 - B_LOO
x_rho   = V_t - V_0
y ~= alpha * x_alpha + rho * x_rho
```

Rows are weighted by trainable-token count because the actor loss is
token-normalized. The features and target are divided by the reward-range width
before their moments are accumulated, so `ridge` is invariant to a linear
rescaling of the reward. The fit targets the unclipped linear residual; the
full TETHER baseline is still clipped once when advantages are constructed.
`V_0` is the first native model-sampled action value on the branch. On a
shared-prefix fork, that remains the shared branch start even when the prefix's
policy gradient is deduplicated onto another leaf.

There is one estimator per training environment, shared globally across that
environment's prompts. A group snapshots the current coefficients before any
of its advantages are assigned. Only after every sibling is scored are its
sufficient statistics queued; each exact `batch_size` rollouts produces a new
ridge fit, followed by

```text
beta <- ema_decay * beta + (1 - ema_decay) * beta_batch
```

The EMA is intentionally not bias-corrected: its early shrinkage toward the
zero/zero LOO baseline is the safety ramp. The completed batch can therefore
affect only later groups. Together with LOO and the evaluator's causal
pre-action values, this prevents the coefficient fit from introducing a
same-action reward-dependent baseline when sibling rewards are scored
independently. A joint or rank-based group scorer can make sibling rewards
depend on the current rollout, in which case LOO alone does not provide this
guarantee. Regression collection happens before the policy-batch warmup gate,
so coefficients adapt during value warmup too.

### Position-conditioned TETHER

Position conditioning is opt-in. It replaces the global coefficient pair with
fixed-width bins over causal, branch-local action-token depth:

```toml
[orchestrator.algo.baseline]
type = "tether"
alpha = 1.0
rho = 1.0

[orchestrator.algo.baseline.position]
bin_size = 1024
# max_action_tokens = 20000  # otherwise the policy sequence length
```

The position of an action token is the number of native model-sampled action
tokens before it on that root-to-leaf branch. Prompt tokens, tool results, user
feedback, and rendering scaffold neither receive an advantage nor advance the
position. Shared sampled prefixes advance every descendant branch, while their
gradient/regression rows are still counted only once. Independent branches
start at zero. This definition does not impose a timing-dependent total order
on concurrent subagents.

Bin boundaries and the horizon are fixed before sampling. The implementation
does not normalize by a rollout's realized final length, since doing so would
let future stopping decisions change the baseline of an earlier action. Tokens
beyond an explicit `max_action_tokens` use the final bin. The position config
must resolve to between 2 and 128 bins; the default 1024-token width gives 20
bins for a 20k-token horizon.

In static mode, the configured `alpha` and `rho` are endpoints. For `K` bins,
bin `k` uses the ex-ante ramp

```text
gate_k  = k / (K - 1)
alpha_k = gate_k * alpha
rho_k   = gate_k * rho
```

The first bin is therefore the configured group anchor (LOO under the default
configuration). With `alpha = rho = 1`, the final bin is the pure current value
`V_t`; other endpoint values remain valid static control-variate choices.

Adding the ordinary adaptive table fits one independent alpha/rho pair per
position bin:

```toml
[orchestrator.algo.baseline.adaptive]
ridge = 1e-6
ema_decay = 0.9
initial_alpha = 0.0
initial_rho = 0.0
# min_bin_rollouts = 32
```

All bins start from the adaptive initial coefficients (zero/zero, hence LOO,
by default). A bin is fitted only from rows in the current exact rollout
regression window and only when enough distinct rollouts reached it. The
default support threshold is one eighth of the regression batch, rounded up.
Unsupported bins retain their coefficients without EMA decay. Supported bins
use `ema_decay ** (contributing_rollouts / batch_size)`, so sparse tail evidence
moves a coefficient less than full-batch evidence. Each bin's moments are
normalized by its own trainable-token count, and its ridge strength is scaled
by `batch_size / contributing_rollouts`. A bin's fit is therefore invariant to
unrelated tokens in other bins while sparse bins are regularized more strongly,
without retaining stale raw rollouts across critic versions.

The applied coefficients, raw batch fits, MSE proxies, clipping fraction,
condition number, update count, and pending rollout count are logged on the
wall-clock axis under `algorithm/<env>/tether/*`. Coefficients need not rise
monotonically as the critic improves: a directionally correct but shrunken
critic can need `rho > 1`, while better calibration can later move it back
toward one. Orchestrator checkpoints preserve the EMA and pending sufficient
statistics. Positioned adaptive runs additionally log each bin's coefficients,
raw fit, support, token count, age, condition number, and update counts under
`algorithm/<env>/tether/position/bin_NNN/*`, plus min/max and active-bin
summaries. A run interrupted before its first policy checkpoint has no
warmup-only orchestrator checkpoint to restore.

`value`, `linear_mix`, and `tether` are invalid unless `[value_function]` is
enabled. Enabling a value function with `mean` or `leave_one_out` is valid and
trains the critic for diagnostics or a later baseline change.

## Staleness and overload

Value staleness is independent of policy off-policy level. Each evaluator
response records one `value_version`; a full value-training batch records the
minimum and maximum policy and evaluator versions represented. Evaluator
coherence is enforced within every rollout group used for policy credit, while
mixing independently labeled groups in a critic optimizer batch is allowed.
The orchestrator reports value-batch fill and publication/drop counts. The
trainer reports `value/source_batches_skipped`, derived from source batch-id
gaps, so replacements caused by conflation remain visible even when the
producer's non-blocking send succeeded. It does not cancel policy rollouts when
a new value version appears.

The value queue is intentionally lossy and capacity one. This is the desired
overload behavior for an online regressor: train repeatedly on one full recent
batch, then move to the newest available policy distribution instead of
working through an increasingly stale FIFO backlog.

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
| `value/source_policy_version_{min,max,spread}`, `value/batch_id`, `value/source_batches_skipped` | Policy provenance range and latest-only transport replacement pressure. |
| `value/batch_{tokens,rollouts,samples}`, `value/total_{tokens,samples}` | Per-update and cumulative critic data volume. Samples count branches; rollouts count episodes used to trigger the batch. |
| `value/update_seconds`, `value/tokens_per_second` | Value optimizer performance. |
| `optim/lr`, `optim/grad_norm`, `optim/zero_grad_ratio` | Critic optimizer health. |
| `value/evaluator_{requests,sequences,tokens,errors,error_rate}` | Cumulative evaluator service volume and failures. |
| `value/evaluator_latency_seconds_{mean,max}` | End-to-end HTTP evaluation latency, including dynamic-batcher waiting. |
| `value/evaluator_version`, `value/evaluator_version_spread` | Evaluator versions represented in a policy batch. A nonzero spread is corrected by coherent group re-evaluation before advantages are stamped. |
| `value/rollout_{prediction,advantage,target}_{mean,std,min,max}` | Values used on the actual policy rollouts, before the critic optimizer update. |
| `value/batch_{pending,target}_rollouts`, `value/batches_{published,dropped}`, `value/batch_drop_rate` | Producer-side critic batch fill and latest-only queue pressure. Dropping stale full batches is expected under overload. |
| `algorithm/<env>/tether/{alpha,rho,batch_fit_alpha,batch_fit_rho,batch_fit_valid}` | Adaptive TETHER coefficients currently applied to new groups and the most recent raw ridge fit. `batch_fit_valid=0` marks a skipped singular/non-finite fit. |
| `algorithm/<env>/tether/{mse_loo,mse_batch_fit,mse_ema,clip_fraction,condition_number}` | Token-weighted residual-variance proxies and fit conditioning. `clip_fraction` is the realized rate under the pre-update coefficients that scored those rollouts. Heavy clipping or poor conditioning makes coefficient magnitude less informative. |
| `algorithm/<env>/tether/{updates,skipped_updates,pending_rollouts,regression_batch_size}` | Adaptive fit cadence and health. These continue updating and logging during value warmup. |

## Warmup

`warmup_updates` is a value-version barrier, not a fixed number of rollout
batches. During warmup the dispatcher and policy inference keep generating and
publishing full critic batches, while policy batches are withheld. As soon as the
evaluator has adopted `warmup_updates`, normal policy shipping begins. Loading
a value checkpoint may set the initial version and satisfy some or all of the
barrier.

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
Resuming one never rewrites the other's step. Value evaluator weights are a
serving copy and are reconstructed from the value checkpoint plus subsequent
trainer publications. On normal policy completion, an unlimited value trainer
stops at its next transport poll and writes a final value checkpoint before the
launcher tears down the value plane.

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

## Initial scope

The first implementation supports text models, one RL run, and direct NCCL
value-weight publication. LoRA, VLM inputs, expert-router replay, and
multi-run value sharing fail validation rather than silently behaving
incorrectly. The transport, evaluator client, baseline, and value-model APIs
are separate modules so those capabilities can be added without changing the
policy trainer contract.
