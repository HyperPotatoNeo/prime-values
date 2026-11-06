"""Native taskset fixture recording actual dataset ownership across processes."""

import json
import os
from pathlib import Path

import verifiers.v1 as vf


class OwnershipData(vf.TaskData):
    value_function_prompt: str
    nested: dict[str, list[str]]


class OwnershipTask(vf.Task[OwnershipData]):
    pass


class OwnershipConfig(vf.TasksetConfig):
    load_log: str


class OwnershipTaskset(vf.Taskset[OwnershipTask, OwnershipConfig]):
    def load(self):
        with Path(self.config.load_log).open("a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "event": "load"}) + "\n")
        yield OwnershipTask(
            OwnershipData(
                idx=17, prompt="selected task", value_function_prompt="private answer", nested={"x": ["first"]}
            ),
            self.config.task,
        )


__all__ = ["OwnershipTaskset"]
