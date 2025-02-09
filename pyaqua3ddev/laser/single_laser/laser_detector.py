"""Abstract base class for detecting lasers."""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class LaserDetector(ABC):
    """Abstract base class for detecting lasers."""

    def _correct_laser(
        self, img: np.ndarray[np.uint8], laser_position: np.ndarray[float], tolerance=10
    ) -> np.ndarray[np.uint8]:
        height, width, _ = img.shape

        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(
            img,
            mask,
            laser_position.round().astype(int),
            0,
            tolerance,
            tolerance,
            4
            | cv2.FLOODFILL_FIXED_RANGE
            | cv2.FLOODFILL_MASK_ONLY
            | 255 << 8,  # Fill with 255
        )
        mask = cv2.UMat(mask[1:-1, 1:-1])  # Move to GPU

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("Couldn't correct laser coordinates")
            return laser_position

        c = max(contours, key=cv2.contourArea)

        moments = cv2.moments(c)

        if moments["m00"] == 0:
            print("Couldn't correct laser coordinates")
            return laser_position

        center_x = float(moments["m10"] / moments["m00"])
        center_y = float(moments["m01"] / moments["m00"])

        return np.array([center_x, center_y])

    @abstractmethod
    def find_laser(self, img: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        """Finds the single laser in the specified img.

        Args:
            img (np.ndarray[np.uint8]): The image to find the laser in.

        Returns:
            np.ndarray[np.uint8]: Returns the pixel location of the laser.
        """
        raise NotImplementedError
