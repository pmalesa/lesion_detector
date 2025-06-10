from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image


def load_image(
    path: str,
    image_metadata: pd.DataFrame,
    hu_scale: bool = True,
    norm: bool = True,
    per_image_norm: bool = True,
):
    img = Image.open(path)
    img_array = np.array(img)
    if hu_scale:
        hu_min_str, hu_max_str = image_metadata["DICOM_windows"].split(",")
        hu_min = float(hu_min_str.strip())
        hu_max = float(hu_max_str.strip())
        return convert_to_hu(img_array, norm, hu_min, hu_max)
    elif norm:
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


def normalize(img: NDArray[np.uint16], per_image_norm: bool):
    img = img.astype(np.float32)
    if not per_image_norm:
        return img / 65535.0
    max = np.max(img)
    min = np.min(img)
    img = (img - min) / (max - min)
    return img


def convert_to_hu(img: NDArray[np.uint16], norm: bool, hu_min=-1024, hu_max=3071):
    """
    Converts the pixel data of a uint16
    CT image to Hounsfield Units (HU).
    """

    hu_img = img.astype(np.int32) - 32768
    hu_img = np.clip(hu_img, hu_min, hu_max).astype(np.float32)
    if norm:
        hu_img = (hu_img - hu_min) / (hu_max - hu_min)
        hu_img = np.clip(hu_img, 0.0, 1.0)
    return hu_img


# TODO - there can be multiple rows with the same image name (fix it)!
def get_image_metadata(metadata: pd.DataFrame, image_name: str):
    """
    Returns the dataframe of a single row from the whole
    metadata dataframe, given the image name.
    """

    return metadata.loc[metadata["File_name"] == image_name].iloc[0]


def get_image_names(split_type_str: str, metadata: pd.DataFrame):
    """
    Returns a list of key slices' image names, given the split
    type (train, validation or test).
    """

    split_type = None
    match split_type_str:
        case "train":
            split_type = 1
        case "validation":
            split_type = 2
        case "test":
            split_type = 3
        case _:
            split_type = -1

    image_names = []
    for i in range(len(metadata)):
        if metadata["Train_Val_Test"][i] == split_type:
            image_names.append(metadata["File_name"][i])

    return image_names
