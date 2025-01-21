from abc import ABC, abstractmethod

import cv2
import numpy as np


class LaserDetector(ABC):
    def __init__(self) -> None:
        super().__init__()

    def _correct_laser(
        self, img: np.ndarray[np.uint8], laser_position: np.ndarray[float], tolerance=10
    ) -> np.ndarray[np.uint8] | None:
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
            return None

        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)

        if M["m00"] == 0:
            return None

        cX = float(M["m10"] / M["m00"])
        cY = float(M["m01"] / M["m00"])

        return np.array([cX, cY])

    @abstractmethod
    def find_laser(self, img: np.ndarray[np.uint8]) -> np.ndarray[np.uint8] | None:
        raise NotImplementedError
