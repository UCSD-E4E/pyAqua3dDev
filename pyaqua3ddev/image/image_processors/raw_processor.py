"""An image processor that takes in a RAW file and produces an image."""

from pathlib import Path

import cv2
import numpy as np
import rawpy

# These modules do exist. Pylint can't find them for some reason.
# pylint: disable=no-name-in-module
from skimage.exposure import adjust_gamma, equalize_adapthist

from pyaqua3ddev.image.image_processors.image_processor import ImageProcessor


class RawProcessor(ImageProcessor):
    """An image processor that takes in a RAW file and produces an image."""

    def __init__(self, gamma=0.3) -> None:
        super().__init__()

        self.__gamma = gamma

    def __double_2_uint16(self, img: np.ndarray) -> np.ndarray:
        return (img * 65535).astype(np.uint16)

    def process(self, file: Path) -> np.ndarray:
        with rawpy.imread(file.as_posix()) as raw:
            img = raw.postprocess(
                gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16
            )
            img = adjust_gamma(img, gamma=self.__gamma)
            img = equalize_adapthist(img)

            return cv2.cvtColor(self.__double_2_uint16(img), cv2.COLOR_RGB2BGR)
