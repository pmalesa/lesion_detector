from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_image(path: str, norm: bool = True, per_image_norm: bool = True):
    img = Image.open(path)
    img_array = np.array(img)
    if norm:
        return normalize(img_array, per_image_norm)
    return img_array


def save_image(img_array: NDArray[Any], path: str):
    if img_array.dtype != np.uint16:
        img_array = img_array.astype(np.uint16)
    img = Image.fromarray(img_array)
    img.save(path)


def show_image(img: NDArray[np.float32], title="None", cmap="gray"):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def normalize(img: NDArray[np.uint16], per_image_norm):
    img = img.astype(np.float32)
    if not per_image_norm:
        return img / 65535.0
    max = np.max(img)
    min = np.min(img)
    img = (img - min) / (max - min)
    return img


# # TO REMOVE
# def main():
#     print("START")
#     image_array = load_image("../../data/single_scan/000001_03_01/088.png")
#     show_image(image_array)
#     image_array = normalize(image_array)
#     print(image_array)
#     save_image(image_array * 65535.0, "./test_img.png")

# # TO REMOVE
# if __name__ == "__main__":
#     main()
