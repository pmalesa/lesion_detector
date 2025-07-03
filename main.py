import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import logging
from pathlib import Path

from src.common.file_utils import load_config
from src.localizer.evaluation import evaluate_localizer
from src.localizer.trainers.train_regressor import train_regressor
from src.localizer.trainers.train_rl import (
    run_complete_training,
    train_single_localizer,
)


def main():
    # Parse command line parameters
    parser = argparse.ArgumentParser(description="Lesion Detector Entry Point")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "complete_training",
            "train_localizer",
            "eval_localizer",
            "train_regressor",
            "train_classifier",
            "test_classifier",
        ],
        help="Specify which task to run/",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Path to the trained model directory (required for evaluation)",
    )
    args = parser.parse_args()

    if args.task == "eval_localizer" and not args.run_dir:
        parser.error("--run-dir is required when --task=eval_localizer")

    # Load configuration
    config_path = ""
    if args.task in ["train_localizer", "complete_training", "train_regressor"]:
        config_path = Path("src/configs/localizer_config.yaml")
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
    else:
        config_path = Path(f"{args.run_dir}/config.yaml")
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
    config = load_config(config_path)

    # Set up logging
    logger = logging.getLogger("LESION-DETECTOR")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt=config["logging"]["format"],
        datefmt=config["logging"]["datefmt"],
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Run given task
    logger.info(f"Starting the task: {args.task}")
    if args.task == "train_localizer":
        algorithm = "dqn"  # TODO
        model, run_dir, seed = train_single_localizer(algorithm, config)
        evaluate_localizer(model, algorithm, config, seed, run_dir)
    elif args.task == "complete_training":
        algorithm = "dqn"  # TODO
        run_complete_training(config, algorithm)
    elif args.task == "eval_localizer":
        pass
        # model_weights_path = f"{args.run_dir}/model.keras"
        # evaluate_localizer(config, model_weights_path)
    elif args.task == "train_regressor":
        train_regressor(config)
    elif args.task == "train_classifier":
        pass
    elif args.task == "test_classifier":
        pass
    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)

    logger.info("Task completed successfully.")


if __name__ == "__main__":
    main()
