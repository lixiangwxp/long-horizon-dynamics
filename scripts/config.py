import argparse


FULL_STATE_DATASETS = ["neurobemfullstate", "pitcnfullstate", "nanodronefullstate"]
MODEL_TYPES = ["mlp", "lstm", "gru", "tcn", "tcnlstm", "grutcn"]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def int_list(value):
    if isinstance(value, list):
        return value
    return [int(element) for element in str(value).split(",") if element != ""]


def int_or_float(value):
    value = str(value)
    if "." in value:
        return float(value)
    return int(value)


def parse_args():
    print("Parsing arguments ...")
    parser = argparse.ArgumentParser(description="Arguments for dynamics learning")

    parser.add_argument(
        "-N", "--model_type", type=str, default="gru", choices=MODEL_TYPES
    )
    parser.add_argument("--encoder_sizes", type=int_list, default=[256])
    parser.add_argument("--decoder_sizes", type=int_list, default=[64, 64, 32])
    parser.add_argument("--encoder_output", type=str, default="output")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("-r", "--run_id", type=int, default=1)
    parser.add_argument("-d", "--gpu_id", type=int, default=0)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("-s", "--shuffle", type=bool, default=False)
    parser.add_argument("-n", "--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--predictor_type", type=str, default="full_state", choices=["full_state"]
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )

    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--cosine_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--cosine_steps", type=int, default=30000)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-08)

    parser.add_argument("--lambda_p", type=float, default=1.0)
    parser.add_argument("--lambda_v", type=float, default=1.0)
    parser.add_argument("--lambda_q", type=float, default=1.0)
    parser.add_argument("--lambda_omega", type=float, default=1.0)

    parser.add_argument("-p", "--plot", type=bool, default=False)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--plot_freq", type=int, default=20)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=int_or_float, default=1.0)
    parser.add_argument("--limit_val_batches", type=int_or_float, default=1.0)
    parser.add_argument("--limit_predict_batches", type=int_or_float, default=0)
    parser.add_argument("--early_stopping", type=str_to_bool, default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=300)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-5)

    parser.add_argument("--sampling_frequency", type=int, default=100)
    parser.add_argument("--unroll_length", type=int, default=2)
    parser.add_argument("--history_length", type=int, default=20)
    parser.add_argument("--delta", type=bool, default=True)
    parser.add_argument(
        "--dataset", type=str, default="neurobemfullstate", choices=FULL_STATE_DATASETS
    )
    parser.add_argument(
        "--nanodrone_raw_path",
        type=str,
        default="/Users/lixiang/Developer/nanodroneclone",
    )
    parser.add_argument("--eval_horizons", type=str, default="1,10,25,50")
    parser.add_argument("--experiment_path", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    return parser.parse_args()


def save_args(args, file_path):
    print("Saving arguments ...")
    with open(file_path, "w") as file:
        for arg in vars(args):
            arg_value = getattr(args, arg)
            arg_type = str(type(arg_value)).replace("<class '", "")[:-2]
            file.write(arg)
            file.write(";")
            file.write(arg_type)
            file.write(";")
            file.write(str(arg_value))
            file.write("\n")


def load_args(file_path):
    print("Loading arguments ...")
    parser = argparse.ArgumentParser(description="Arguments for dynamics learning")
    with open(file_path, "r") as file:
        for line in file.readlines():
            arg_name, arg_type, raw_value = line.rstrip("\n").split(";", maxsplit=2)
            if arg_type == "str":
                parser.add_argument("--" + arg_name, type=str, default=raw_value)
            elif arg_type == "int":
                parser.add_argument("--" + arg_name, type=int, default=int(raw_value))
            elif arg_type == "float":
                parser.add_argument(
                    "--" + arg_name, type=float, default=float(raw_value)
                )
            elif arg_type == "list":
                value = raw_value.strip()
                if value == "[]":
                    parsed_value = []
                else:
                    parsed_value = [int(element) for element in value[1:-1].split(", ")]
                parser.add_argument("--" + arg_name, type=list, default=parsed_value)
            elif arg_type == "tuple":
                value = raw_value.strip()
                if value == "()":
                    parsed_value = ()
                else:
                    parsed_value = tuple(
                        int(element) for element in value[1:-1].split(", ")
                    )
                parser.add_argument("--" + arg_name, type=tuple, default=parsed_value)
            elif arg_type == "bool":
                parsed_value = raw_value == "True"
                parser.add_argument("--" + arg_name, type=bool, default=parsed_value)

    loaded_args = parser.parse_args([])

    if getattr(loaded_args, "predictor_type", None) in {"velocity", "attitude"}:
        raise ValueError(
            "velocity/attitude Rao predictors are not compatible with the current "
            "full-state pipeline. Use --predictor_type full_state."
        )

    return loaded_args
