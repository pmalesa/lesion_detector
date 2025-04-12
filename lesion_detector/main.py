import argparse
import logging
import sys
from pathlib import Path

from common.file_utils import load_config
from localizer.trainer import evaluate_localizer, train_localizer


def main():
    # Parse command line parameters
    parser = argparse.ArgumentParser(description="Lesion Detector Entry Point")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "train_localizer",
            "eval_localizer",
            "train_classifier",
            "test_classifier",
        ],
        help="Specify which task to run/",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=False,
        help="Path to the trained model directory (required for evaluation)",
    )
    args = parser.parse_args()

    if args.task == "eval_localizer" and not args.run_dir:
        parser.error("--run-dir is required when --task=eval_localizer")

    # Load configuration
    config_path = ""
    if args.task == "train_localizer":
        config_path = Path("configs/localizer_config.yaml")
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
        train_localizer(config)
    elif args.task == "eval_localizer":
        model_weights_path = f"{args.run_dir}/model.keras"
        evaluate_localizer(config, model_weights_path)
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
