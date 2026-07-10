# Value functions

Value functions are an optional, independent data plane in an RL run. They do
not add a value loss to the policy trainer. The policy trainer receives the
same token ids, sampling log probabilities, masks, and already-computed
advantages as it does without a critic.

## Runtime topology

Enabling `[value_function]` adds two roles:

1. The **value trainer** receives completed trajectories on a bounded,
   latest-only queue. It runs `updates_per_batch` optimizer updates on the
   newest available batch, discards that batch, and publishes a monotonically
   increasing value version after every optimizer update.
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
4. sends lambda returns to the value trainer while stamping the selected
   advantage on the policy sample.

Context, prompt, and tool-response tokens are never value-loss members. Packed
sequences reset both the causal value shift and GAE recursion at every sequence
boundary. Value predictions are versioned; if an update lands while siblings
are being scored, the whole group is re-evaluated at one coherent version.

The default `gamma = 1` and `gae_lambda = 1` gives Monte Carlo return-to-go
targets. This also makes warmup independent of the randomly initialized value
head.

## Value head and losses

The default loss is scalar MSE regression. The value head is one unconstrained
linear output: there is no sigmoid or other bounding activation.

Classification is optional. It uses `num_bins` logits over evenly spaced
support points in `reward_range`; softmax expectation gives the scalar value.
Continuous targets are projected onto their two adjacent support bins, which
preserves the target expectation instead of rounding it to one class. Targets
outside the configured range fail immediately. The RG-Mix example uses two
bins over `[0, 1]`, so binary rewards remain ordinary one-hot classes.

## Baselines

GRPO keeps the existing group-mean baseline by default. Its `baseline` is a
discriminated configuration with non-value and value-backed choices:

- `mean`: standard GRPO, `A = R - mean(R)`;
- `leave_one_out`: `A_i = R_i - mean(R_{j != i})`;
- `value`: pure per-token GAE;
- `linear_mix`: a convex blend of a group advantage and GAE;
- `tether`: a clipped two-factor correction anchored on a group baseline.

Both `linear_mix` and `tether` accept `group = "mean"` or
`group = "leave_one_out"`; **leave-one-out is the default for both**.

TETHER forms the complete baseline and clips once at the end:

```text
b_t = clip(B_group + alpha * (V_0 - B_group) + rho_t * (V_t - V_0), low, high)
A_t = R - b_t
```

The anchor correction and progress term are not clipped separately.

`value`, `linear_mix`, and `tether` are invalid unless `[value_function]` is
enabled. Enabling a value function with `mean` or `leave_one_out` is valid and
trains the critic for diagnostics or a later baseline change.

## Staleness and overload

Value staleness is independent of policy off-policy level. Each evaluator
response and each value-training batch records `value_version`. The
orchestrator reports the evaluator version and value-batch publication/drop
counts. The trainer reports `value/source_batches_skipped`, derived from source
batch-id gaps, so replacements caused by conflation remain visible even when
the producer's non-blocking send succeeded. It does not cancel policy rollouts
when a new value version appears.

The value queue is intentionally lossy and capacity one. This is the desired
overload behavior for an online regressor: train repeatedly on one coherent
recent batch, then move to the newest available policy distribution instead of
working through an increasingly stale FIFO backlog.

## Warmup

`warmup_updates` is a value-version barrier, not a fixed number of rollout
batches. During warmup the dispatcher and policy inference keep generating and
publishing trajectories, while policy batches are withheld. As soon as the
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
