import matplotlib.pyplot as plt
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

max_width = 0
max_height = 0
max_aspect_ratio = 0.0
count = 0
ar_count = 0

max_width_img_name = ""
max_height_img_name = ""
max_ar_img_name = ""

for i in range(len(metadata)):
    bbox_str = metadata["Bounding_boxes"][i]
    img_name = metadata["File_name"][i]
    lesion_location_str = metadata["Normalized_lesion_location"][i]
    lesion_diameters_str = metadata["Lesion_diameters_Pixel_"][i]

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
    widths.append(w)
    heights.append(h)

    if max_width < w:
        max_width = w
        max_width_img_name = img_name

    if max_height < h:
        max_height = h
        max_height_img_name = img_name

    if w > 64 and h > 64:
        count += 1

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
print(f"MAX WIDTH: {max_width}")
print(f"MAX HEIGHT: {max_height}")
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
