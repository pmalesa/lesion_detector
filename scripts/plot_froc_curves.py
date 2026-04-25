from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_NAMES = [
    "faster_rcnn",
    "yolo",
    "retinanet",
    "detr",
    "deformable_detr",
    "deformable_detr+",
    "deformable_detr++",
]

MODEL_LABELS = {
    "faster_rcnn": "Faster R-CNN",
    "yolo": "YOLOv8",
    "retinanet": "RetinaNet",
    "detr": "DETR",
    "deformable_detr": "Deformable DETR",
    "deformable_detr+": "Deformable DETR +",
    "deformable_detr++": "Deformable DETR ++",
}

COLORS = {
    "faster_rcnn": "tab:blue",
    "yolo": "tab:orange",
    "retinanet": "tab:green",
    "detr": "tab:red",
    "deformable_detr": "tab:purple",
    "deformable_detr+": "tab:brown",
    "deformable_detr++": "tab:pink",
}


def load_froc_file(file_path):
    df = pd.read_csv(file_path, sep=r"\s+")
    return df["fp_per_image"].values, df["sensitivity"].values


def plot_froc_curves(froc_dir, title, output_path=None):
    froc_dir = Path(froc_dir)

    plt.figure(figsize=(9, 5))

    for model_name in MODEL_NAMES:
        file_path = froc_dir / f"{model_name}_froc_values.txt"

        if not file_path.exists():
            print(f"Warning: missing file: {file_path}")
            continue

        fp, sens = load_froc_file(file_path)

        plt.plot(
            fp,
            sens,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            linewidth=2.5,
        )

    plt.xlabel("Liczba fałszywych detekcji na obraz", fontsize=20)
    plt.ylabel("Czułość", fontsize=20)
    plt.title(title, fontsize=20)

    # Tick label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(0, 30)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.2)
    plt.legend()

    # Legend font size
    plt.legend(fontsize=18)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


# Paths
plot_froc_curves(
    froc_dir="../results/froc/deeplesion",
    title="Krzywe FROC - DeepLesion",
    output_path="deeplesion_froc_curves.png",
)

plot_froc_curves(
    froc_dir="../results/froc/kidney_stones",
    title="Krzywe FROC - Kidney Stones",
    output_path="kidney_stones_froc_curves.png",
)
