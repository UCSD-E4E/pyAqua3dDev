"""A laser detector that uses labels from Label Studio."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from pyaqua3ddev.laser.single_laser.laser_detector import LaserDetector


class LabelStudioLaserDetector(LaserDetector):
    """A laser detector that uses labels from Label Studio."""

    def __init__(self, image_path: Path, label_studio_json_path: Path):
        super().__init__()

        self.__image_path = image_path
        self.__label_studio_json_path = label_studio_json_path
        self.__hash = hashlib.md5(image_path.read_bytes()).hexdigest()

    def find_laser(self, img: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        if not self.__label_studio_json_path.exists():
            raise IOError

        with self.__label_studio_json_path.open("r") as f:
            label_studio = json.load(f)

        for item in label_studio:
            path_string: str = item["data"]["img"]

            if path_string.startswith("https://e4e-nas.ucsd.edu:6021"):
                url = urlparse(path_string)

                if (
                    Path(url.path).stem != self.__image_path.stem
                    and Path(url.path).stem != self.__hash
                ):
                    continue
            elif path_string.startswith("/"):
                path = Path(path_string)

                if path.stem != self.__hash:
                    continue
            else:
                raise NotImplementedError

            if len(item["annotations"]) == 0:
                continue

            if len(item["annotations"][0]["result"]) == 0:
                return None

            result = item["annotations"][0]["result"][0]
            original_width = float(result["original_width"])
            original_height = float(result["original_height"])

            x = result["value"]["x"] * original_width / 100.0
            y = result["value"]["y"] * original_height / 100.0

            if original_width == 3987:
                y -= 10

            return np.array([x, y])  # self._correct_laser(img, np.array([x, y]))

        print(f"Couldn't find image {self.__image_path}")
        return None
