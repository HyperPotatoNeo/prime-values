#!/usr/bin/env bash
# Run only in a dedicated test environment: tests/conftest.py cleans up torchrun processes.
set -euo pipefail
result_dir=${1:?usage: run_checks.sh RESULT_DIRECTORY}
mkdir -p "$result_dir"
uv run --no-sync python - "$result_dir/runtime.json" <<'PY'
import importlib.metadata as md
import hashlib
import json
import platform
import sys
import torch
import prime_rl.value.math as value_math
import prime_rl.configs.value as value_config
from pathlib import Path
root = Path.cwd().resolve()
sources = {'value_math': Path(value_math.__file__).resolve(), 'value_config': Path(value_config.__file__).resolve()}
assert sources['value_math'] == root / 'src/prime_rl/value/math.py', sources
assert sources['value_config'] == root / 'packages/prime-rl-configs/src/prime_rl/configs/value.py', sources
packages = ['torch', 'transformers', 'vllm', 'verifiers', 'renderers', 'flash-attn', 'pytest']
Path(sys.argv[1]).write_text(json.dumps({
    'python': platform.python_version(),
    'packages': {name: md.version(name) for name in packages},
    'cuda': torch.version.cuda,
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'imported_sources': {name: str(path) for name, path in sources.items()},
    'file_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted([root / 'uv.lock', root / 'pyproject.toml',
                                        *root.glob('src/**/*.py'), *root.glob('tests/**/*.py'),
                                        *root.glob('packages/**/*.py')])},
}, indent=2) + '\n')
PY
uv run --no-sync pytest -q \
  tests/unit/value \
  tests/unit/orchestrator/test_advantage.py \
  tests/unit/orchestrator/test_algorithms.py \
  tests/unit/orchestrator/test_adaptive_baselines.py \
  tests/unit/orchestrator/test_train_sink.py \
  tests/unit/orchestrator/test_orchestrator_setup.py \
  tests/unit/orchestrator/test_value_context.py \
  tests/unit/orchestrator/test_value_context_renderers.py \
  tests/unit/test_adaptive_baseline_configs.py \
  tests/unit/test_configs.py \
  tests/unit/train/rl/test_loss.py \
  --junitxml="$result_dir/pytest.xml" 2>&1 | tee "$result_dir/pytest.log"
uv run --no-sync python - "$result_dir/pytest.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).getroot().iter('testcase'))
assert cases, 'No test cases ran'
assert not any(case.find(tag) is not None for case in cases for tag in ('skipped', 'error', 'failure'))
print(f'Verified {len(cases)} cases without skips, errors or failures')
PY
