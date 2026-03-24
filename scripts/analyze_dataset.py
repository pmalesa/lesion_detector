import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "../data/deeplesion_metadata.csv"
metadata = pd.read_csv(path)
n_images = len(metadata)


bboxes: list = []
aspect_ratios: list = []
widths: list = []
heights: list = []
long_diameters: list = []
short_diameters: list = []
locations_x: list = []
locations_y: list = []
lesion_counts: dict[int, int] = {}

class_names = [
    "bone",
    "abdomen",
    "mediastinum",
    "liver",
    "lung",
    "kidney",
    "soft_tissue",
    "pelvis",
]
lesion_name_map = {
    1: "kość",
    2: "jama brzuszna",
    3: "śródpiersie",
    4: "wątroba",
    5: "płuca",
    6: "nerka",
    7: "tkanka miękka",
    8: "miednica",
}

max_width = 0
max_height = 0
max_aspect_ratio = 0.0
count = 0
ar_count = 0

max_width_img_name = ""
max_height_img_name = ""
max_ar_img_name = ""

for i in range(len(metadata)):
    # Analyze only annotated data
    if metadata["Train_Val_Test"][i] == 1:
        continue
    bbox_str = metadata["Bounding_boxes"][i]
    img_name = metadata["File_name"][i]
    lesion_location_str = metadata["Normalized_lesion_location"][i]
    lesion_diameters_str = metadata["Lesion_diameters_Pixel_"][i]

    lesion_type = metadata["Coarse_lesion_type"][i]
    lesion_counts[lesion_type] = lesion_counts.get(lesion_type, 0) + 1

    # Get mm spacing per pixel
    pixel_spacing_str = metadata["Spacing_mm_px_"][i]
    pixel_spacing = [float(val) for val in pixel_spacing_str.split(",")]
    pixel_spacing_x = pixel_spacing[0]
    pixel_spacing_y = pixel_spacing[1]

    # Process lesion locations
    lesion_location = [float(val) for val in lesion_location_str.split(",")]
    lesion_x, lesion_y, lesion_z = [c for c in lesion_location]
    locations_x.append(lesion_x)
    locations_y.append(lesion_y)

    # Process lesion diameters
    lesion_diameters = [float(val) for val in lesion_diameters_str.split(",")]
    long, short = [c for c in lesion_diameters]
    long_diameters.append(long)
    short_diameters.append(short)

    # Process bbox parameters
    bbox_coords = [float(val) for val in bbox_str.split(",")]
    x1, y1, x2, y2 = [int(c) for c in bbox_coords]
    w, h = x2 - x1, y2 - y1

    w_mm = int(w * pixel_spacing_x)
    h_mm = int(h * pixel_spacing_y)
    widths.append(w_mm)
    heights.append(h_mm)

    if max_width < w_mm:
        max_width = w_mm
        max_width_img_name = img_name

    if max_height < h_mm:
        max_height = h_mm
        max_height_img_name = img_name

    # Calculate aspect ratio
    aspect_ratio = w / h if w > h else h / w
    aspect_ratios.append(aspect_ratio)
    if max_aspect_ratio < aspect_ratio:
        max_aspect_ratio = aspect_ratio
        max_ar_img_name = img_name

    if aspect_ratio > 3:
        ar_count += 1

print("----------STATS----------\n")
print(f"NUMBER OF IMAGES: {n_images}")
print(f"MAX WIDTH: {max_width} [mm]")
print(f"MAX HEIGHT: {max_height} [mm]")
print(f"MAX ASPECT RATIO: {max_aspect_ratio} : 1")
print(
    f"Number of bboxes with height and width > 64 -> {count}, \
    {round((count / n_images) * 100, 2)}% of the whole dataset."
)
print(
    f"Number of bboxes with aspect ratio > 3:1 -> {ar_count}, \
    {round((ar_count / n_images) * 100, 2)}% of the whole dataset."
)

print(f"{max_width_img_name}")
print(f"{max_height_img_name}")
print(f"{max_ar_img_name}")

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan"]

fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.ravel()

# 1) Lesion location X vs Y
axs[0].scatter(locations_x, locations_y, s=10, color=colors[0])
axs[0].set_title("Lesion Location X vs Y")
axs[0].set_xlabel("Lesion X location")
axs[0].set_ylabel("Lesion Y location")

# 2) Long diameter vs short diameter
axs[1].scatter(long_diameters, short_diameters, s=10, color=colors[1])
axs[1].set_title("Long Diameter vs Short Diameter")
axs[1].set_xlabel("Long Diameter")
axs[1].set_ylabel("Short Diameter")

# 3) BBox widths vs heights
axs[2].scatter(widths, heights, s=10, color=colors[2])
axs[2].set_title("Bounding Box Width vs Height")
axs[2].set_xlabel("Width")
axs[2].set_ylabel("Height")

# 4) Histogram of BBox aspect ratios
axs[3].hist(aspect_ratios, bins=100, color=colors[3])
axs[3].set_title("Histogram of BBox Aspect Ratios")
axs[3].set_xlabel("Aspect Ratio (larger_dim / smaller_dim)")
axs[3].set_ylabel("Count")

# 5) Histogram of BBox widths
axs[4].hist(widths, bins=100, color=colors[4])
axs[4].set_title("Histogram of BBox Widths")
axs[4].set_xlabel("Width")
axs[4].set_ylabel("Count")

# 6) Histogram of BBox heights
axs[5].hist(heights, bins=100, color=colors[5])
axs[5].set_title("Histogram of BBox Heights")
axs[5].set_xlabel("Height")
axs[5].set_ylabel("Count")

plt.tight_layout()
plt.show()

# ================================
# CLASS IMBALANCE HISTOGRAM
# ================================

TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
ANNOTATION_SIZE = 13

# Sort by frequency (descending)
sorted_items = sorted(lesion_counts.items(), key=lambda x: x[1], reverse=True)

classes_en = [k for k, _ in sorted_items]
counts = [v for _, v in sorted_items]

classes_pl = [lesion_name_map[c] for c in classes_en]

total = sum(counts)
percentages = [c / total * 100 for c in counts]

# Explicit categorical positions
x_pos = np.arange(len(classes_pl))

colors = plt.cm.Set1(np.linspace(0, 1, len(classes_pl)))

plt.figure(figsize=(10, 6))

bars = plt.bar(x_pos, counts, color=colors, edgecolor="black", linewidth=1.2)

plt.ylim(0, 3000)

plt.xticks(x_pos, classes_pl, rotation=30, ha="right", fontsize=TICK_SIZE)

plt.yticks(fontsize=TICK_SIZE)

for bar, count, pct in zip(bars, counts, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_SIZE,
    )

plt.title("Rozkład typów zmian chorobowych", fontsize=TITLE_SIZE, fontweight="bold")

plt.xlabel("Typ zmiany chorobowej", fontsize=LABEL_SIZE)

plt.ylabel("Liczba zmian", fontsize=LABEL_SIZE)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


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
    "Rozkład szerokości i wysokości ramek ograniczających",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Długość [mm]", fontsize=14)

plt.ylabel("Liczba", fontsize=14)

plt.xticks(np.arange(0, max_dim + 10, 10), fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.xlim(0, 100)

plt.tight_layout()
plt.show()
