import torch
from torch.utils.data import Dataset
import cv2
from common.file_utils import extract_filename 
from common.image_utils import load_image, get_image_names, create_image_paths, get_image_metadata
import pandas as pd

class LesionDataset(Dataset):
    """
    A dataset that returns (img_tensor, bbox_tensor) pairs
    for bounding box regression training.
    """

    def __init__(self, split: str, metadata: pd.DataFrame, images_dir: str):
        """
        image_paths: list of filesystem paths to the CT slices
        metadata:    dict or DataFrame mapping filename to bbox coords
        """

        self._split = split
        self._metadata = metadata

        image_names = get_image_names(split, metadata)
        self._image_paths = create_image_paths(image_names, images_dir)


    def __len__(self):
        return len(self._image_paths)
    
    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image_name = extract_filename(image_path)
        image_metadata = get_image_metadata(self._metadata, image_name)

        # Load image and convert to single-channel tensor [1, H, W]
        image_data = load_image(image_path, norm=True)
        image_height = image_data.shape[0]
        image_width = image_data.shape[1]

        # Lookup and rescale the ground truth bbox
        bbox_coords = [float(v) for v in image_metadata["Bounding_boxes"].split(",")]
        x1, y1, x2, y2 = [round(c) for c in bbox_coords]

        # Rescale to (512 x 512) if necessary
        if (image_height, image_width) != (512, 512):
            image_data = cv2.resize(
                image_data, (512, 512), interpolation=cv2.INTER_AREA
            )
            scale_x = 512 / image_width
            scale_y = 512 / image_height
            x1 = round(x1 * scale_x)
            y1 = round(y1 * scale_y)
            x2 = round(x2 * scale_x)
            y2 = round(y2 * scale_y)

        image_data = torch.from_numpy(image_data).unsqueeze(0)
        bbox = torch.tensor([x1, y1, x2 - x1, y2 - y1], dtype=torch.float32)

        return image_data, bbox





