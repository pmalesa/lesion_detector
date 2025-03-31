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
            "test_localizer",
            "train_classifier",
            "test_classifier",
        ],
        help="Specify which task to run/",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = ""
    if "localizer" in args.task:
        config_path = "configs/localizer_config.yaml"
    else:
        config_path = "configs/classifier_config.yaml"
    config_path = Path(config_path)
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
    elif args.task == "test_localizer":
        evaluate_localizer(config)
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
