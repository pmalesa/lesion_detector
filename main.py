import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import logging
from pathlib import Path

from src.common.file_utils import extract_directory, load_config
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
        "--algorithm", type=str, required=True, help="Name of the algorithm (dqn/ppo)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to the trained model directory (required for evaluation)",
    )
    args = parser.parse_args()

    if args.task == "eval_localizer" and not args.model_path:
        parser.error("--model-path is required when --task=eval_localizer")

    # Load configuration
    config_path = ""
    if args.task in ["train_localizer", "complete_training", "train_regressor"]:
        config_path = Path("src/configs/localizer_config.yaml")
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)
    else:
        run_dir = extract_directory(args.model_path)
        config_path = Path(f"{run_dir}/config.yaml")
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
        model_path, seed = train_single_localizer(args.algorithm, config)
        evaluate_localizer(model_path, args.algorithm, config, seed)
    elif args.task == "complete_training":
        run_complete_training(config, args.algorithm)
    elif args.task == "eval_localizer":
        evaluate_localizer(args.model_path, args.algorithm, config, 42)
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
