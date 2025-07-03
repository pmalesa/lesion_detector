import os
from collections import OrderedDict

import numpy as np
from common.image_utils import load_image
from numpy.typing import NDArray


class ImageCache:
    def __init__(self, data_dir: str, maxsize: int = 1000):
        self._data_dir = data_dir
        self._cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, image_name: str):
        if image_name in self._cache:
            self._cache.move_to_end(image_name, last=True)
            return self._cache[image_name]

        image_path = os.path.join(self._data_dir, image_name)
        image = load_image(image_path, norm=True)

        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        self._cache[image_name] = image
        return image
