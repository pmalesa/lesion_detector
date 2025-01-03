import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path)
    img_array = np.array(img)
    return img_array


def save_image(img_array, path):
    if img_array.dtype != np.uint16:
        img_array = img_array.astype(np.uint16)
    img = Image.fromarray(img_array)
    img.save(path)


def show_image(img, title="None", cmap="gray"):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def convert_to_hu(img):
    hu_img = img - 32768
    return hu_img


def normalize(img):
    img = img.astype(np.float32)
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
