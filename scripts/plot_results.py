import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(csv_file: str):

    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    axes[0].plot(df["episode"], df["total_reward"], color=colors[0])
    axes[0].set_title("Total Reward")

    axes[1].plot(df["episode"], df["avg_loss"], color=colors[1])
    axes[1].set_title("Average Loss")

    axes[2].plot(df["episode"], df["iou"], color=colors[2])
    axes[2].set_title("Final IoU")

    axes[3].plot(df["episode"], df["distance"], color=colors[3])
    axes[3].set_title("Final Normalized Distance")

    for axis in axes:
        axis.set_xlabel("Episode")
        axis.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV file")
    parser.add_argument(
        "--csv-file", type=str, required=True, help="Path to training log CSV file"
    )
    args = parser.parse_args()
    if os.path.exists(args.csv_file):
        plot_metrics(args.csv_file)
    else:
        print(f"[ERROR] File not found: {args.csv_file}")
