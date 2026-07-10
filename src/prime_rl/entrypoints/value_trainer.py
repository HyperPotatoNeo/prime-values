def main() -> None:
    # Keep CLI help lightweight; import the distributed implementation only
    # after configuration parsing has selected this entry point.
    from prime_rl.configs.value import ValueFunctionConfig
    from prime_rl.utils.config import cli
    from prime_rl.value.train import train_value

    train_value(cli(ValueFunctionConfig))


if __name__ == "__main__":
    main()
