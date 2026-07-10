import os

from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.value.evaluator import ValueEvaluatorRuntime, ValueEvaluatorServer


def main() -> None:
    set_proc_title("ValueEvaluator")
    config = cli(ValueFunctionConfig)
    setup_logger(config.log.level, json_logging=config.log.json_logging)
    evaluator_rank = int(os.environ.get("VALUE_EVALUATOR_RANK", "0"))
    runtime = ValueEvaluatorRuntime(config, evaluator_rank)
    port = int(os.environ.get("VALUE_EVALUATOR_PORT", str(config.evaluator.port)))
    server = ValueEvaluatorServer((config.evaluator.host, port), runtime)
    server.serve_forever()


if __name__ == "__main__":
    main()
