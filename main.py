import argparse
import os
import sys
from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
    get_out_file,
)
from src.configs import *
from src.thread.run_thread import run_thread
from src.thread.model_eval import run_eval
from src.thread.label_eval.check_human_labels import run_eval_labels
from src.thread.generate_online_profiles import gen_style_thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/thread/thread.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    f, path = get_out_file(cfg)

    try:
        print(cfg)
        if cfg.task == Task.THREAD:
            run_thread(cfg)
        elif cfg.task == Task.GENSTYLETHREAD:
            gen_style_thread(cfg)
        elif cfg.task == Task.EVAL:
            run_eval(cfg)
        elif cfg.task == Task.EVALLabels:
            run_eval_labels(cfg)

        else:
            raise NotImplementedError(f"Task {cfg.task} not implemented")

    except ValueError as e:
        sys.stderr.write(f"Error: {e}")
    finally:
        if cfg.store:
            f.close()
