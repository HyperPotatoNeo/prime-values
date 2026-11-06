"""Sharded EMA of published BF16 policy weights, checkpointed with the trainer."""

import torch
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict


class EMAReference:
    def __init__(self, reference, policy, decay: float):
        self.model = reference
        self.decay = decay
        self.updates = 0
        self.last_publication = -1
        self.model.requires_grad_(False)
        self.model.eval()
        self._copy(policy, 0.0)

    @torch.no_grad()
    def _copy(self, policy, decay):
        target = dict(self.model.named_parameters())
        source = dict(policy.named_parameters())
        if target.keys() != source.keys():
            raise ValueError("EMA reference parameter topology differs from policy")
        for name, dest in target.items():
            # Match the BF16 publication stream; accumulate EMA in the unchanged
            # optimization dtype (FP32 in the research configs).
            value = source[name].detach().to(torch.bfloat16).to(dest.dtype)
            if decay == 0:
                dest.copy_(value)
            else:
                dest.mul_(decay).add_(value, alpha=1 - decay)
        for name, dest in self.model.named_buffers():
            dest.copy_(dict(policy.named_buffers())[name])

    def update(self, policy, publication: int):
        if publication <= self.last_publication:
            raise ValueError("EMA publications must be strictly increasing")
        self._copy(policy, self.decay)
        self.updates += 1
        self.last_publication = publication

    def state_dict(self):
        return {
            "model": get_model_state_dict(self.model),
            "updates": self.updates,
            "last_publication": self.last_publication,
            "decay": self.decay,
        }

    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise ValueError("EMA decay cannot change silently on resume")
        set_model_state_dict(self.model, state["model"])
        self.updates = state["updates"]
        self.last_publication = state["last_publication"]
