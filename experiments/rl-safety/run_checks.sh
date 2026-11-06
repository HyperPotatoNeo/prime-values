#!/usr/bin/env bash
# Use an isolated test pod: pytest's cleanup fixture terminates training processes.
set -euo pipefail
result_dir=${1:?usage: run_checks.sh NEW_RESULT_DIRECTORY}
if [[ -e "$result_dir" ]]; then
  echo "Use a fresh result directory: $result_dir" >&2
  exit 1
fi
bash experiments/value-baseline/run_checks.sh "$result_dir/baseline"
uv run --no-sync pytest -q \
  tests/unit/orchestrator/test_ckpt.py \
  tests/unit/train/test_conversion_cache.py \
  tests/unit/train/test_tensor_summaries.py \
  tests/unit/train/rl/test_normalization.py \
  --junitxml="$result_dir/reliability.xml" 2>&1 | tee "$result_dir/reliability.log"
uv run --no-sync python - "$result_dir/reliability.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).getroot().iter('testcase'))
assert cases and not any(case.find(tag) is not None for case in cases for tag in ('skipped', 'error', 'failure'))
print(f'Verified {len(cases)} reliability cases without skips, errors or failures')
PY
uv run --no-sync python experiments/value-baseline/check_mutations.py "$result_dir/mutations"
