import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from pyaqua3ddev.laser.single_laser.laser_detector import LaserDetector


class LabelStudioLaserDetector(LaserDetector):
    def __init__(self, image_path: Path, label_studio_json_path: Path):
        super().__init__()

        self.__image_path = image_path
        self.__label_studio_json_path = label_studio_json_path

    def find_laser(self, img: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        if not self.__label_studio_json_path.exists():
            raise IOError

        with self.__label_studio_json_path.open("r") as f:
            label_studio = json.load(f)

        for item in label_studio:
            path_string: str = item["data"]["img"]

            if path_string.startswith("https://e4e-nas.ucsd.edu:6021"):
                url = urlparse(url)

                if Path(url.path).stem != self.__image_path.stem:
                    continue

                if len(item["annotations"]) == 0:
                    continue

                if len(item["annotations"][0]["result"]) == 0:
                    return None

                result = item["annotations"][0]["result"][0]
                original_width = result["original_width"]
                original_height = result["original_height"]

                x = result["value"]["x"] * original_width
                y = result["value"]["y"] * original_height

                return self._correct_laser(img, np.array([x, y]))
            else:
                raise NotImplementedError

        raise KeyError("laser label cannot be found.")
