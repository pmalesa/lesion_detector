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

# 1) Lesion location x vs y
plt.figure()
plt.scatter(locations_x, locations_y, s=10)
plt.title("Lesion Location X vs Y")
plt.xlabel("Lesion X location")
plt.ylabel("Lesion Y location")
plt.show()

# 2) Long diameter vs short diameter
plt.figure()
plt.scatter(long_diameters, short_diameters, s=10)
plt.title("Long Diameter vs Short Diameter")
plt.xlabel("Long Diameter")
plt.ylabel("Short Diameter")
plt.show()

# 3) Bbox widths vs heights
plt.figure()
plt.scatter(widths, heights, s=10)
plt.title("Bounding Box Width vs Height")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

# 4) Histogram of bounding box aspect ratios
plt.figure()
plt.title("Histogram of BBox Aspect Ratios")
plt.xlabel("Aspect Ratio (larger_dim / smaller_dim)")
plt.ylabel("Count")
plt.hist(aspect_ratios, bins=100)
plt.show()

# 5) Histograms of widths and heights
# a) BBox widths
plt.figure()
plt.title("Histogram of Bounding Box Widths")
plt.xlabel("Width")
plt.ylabel("Count")
plt.hist(widths, bins=100)
plt.show()

# b) BBox heights
plt.figure()
plt.title("Histogram of Bounding Box Heights")
plt.xlabel("Height")
plt.ylabel("Count")
plt.hist(heights, bins=100)
plt.show()
