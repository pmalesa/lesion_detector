import os

import matplotlib.pyplot as plt
import numpy as np

dataset_labels_path = (
    "../notebooks/data/kidney_stones/kidney_stones_processed_uint8/labels/"
)

IMAGE_SIZE = 512

widths = []
heights = []
n_images = 0

for filename in os.listdir(dataset_labels_path):

    if not filename.endswith(".txt"):
        continue

    n_images += 1
    file_path = os.path.join(dataset_labels_path, filename)

    with open(file_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            # Skip malformed rows
            if len(parts) != 5:
                continue

            # YOLO format: class cx cy w h
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # Convert to pixels
            w_px = w_norm * IMAGE_SIZE
            h_px = h_norm * IMAGE_SIZE

            widths.append(w_px)
            heights.append(h_px)

print(f"Number of bounding boxes: {len(widths)}\n Number of images: {n_images}")

# ================================
# STANDALONE GROUPED HISTOGRAM
# ================================

# Choose bin width in mm
bin_width = 5

max_dim = max(max(widths), max(heights))

# Create bin edges
bins = np.arange(0, max_dim + bin_width, bin_width)

# Compute histogram counts
width_counts, _ = np.histogram(widths, bins=bins)
height_counts, _ = np.histogram(heights, bins=bins)

# Compute bin centers
bin_centers = (bins[:-1] + bins[1:]) / 2

# Width of each bar
bar_width = bin_width * 0.4

# Create separate figure
plt.figure(figsize=(10, 6))

plt.bar(
    bin_centers - bar_width / 2,
    width_counts,
    width=bar_width,
    label="Szerokość",
    color="green",
    edgecolor="black",
)

plt.bar(
    bin_centers + bar_width / 2,
    height_counts,
    width=bar_width,
    label="Wysokość",
    color="gold",
    edgecolor="black",
)

plt.title(
    "Rozkład szerokości i wysokości ramek ograniczających w zbiorze kamieni nerkowych",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Długość w pikselach", fontsize=14)

plt.ylabel("Liczba", fontsize=14)

plt.xticks(np.arange(0, max_dim + 10, 10), fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.xlim(0, 100)

plt.tight_layout()
plt.show()
