import hashlib
from pathlib import Path

import numpy as np
import psycopg

from pyaqua3ddev.laser.single_laser.laser_detector import LaserDetector


class PSqlLabelDetector(LaserDetector):
    def __init__(
        self,
        image_path: Path,
        dbname: str,
        user: str,
        password: str,
        host: str,
        port=5432,
    ):
        super().__init__()

        self.__dbname = dbname
        self.__user = user
        self.__password = password
        self.__host = host
        self.__port = port

        self.__hash = hashlib.md5(image_path.read_bytes()).hexdigest()

    def find_laser(self, img: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        with psycopg.connect(
            host=self.__host,
            user=self.__user,
            password=self.__password,
            dbname=self.__dbname,
            port=self.__port,
        ) as conn:
            cursor = conn.cursor()
            rows = list(
                cursor.execute(
                    """
SELECT laser_labels.x, laser_labels.y FROM laser_labels
INNER JOIN images ON laser_labels.cksum = images.image_md5
INNER JOIN canonical_dives on images.dive = canonical_dives.path
WHERE laser_labels.cksum = %s""",
                    (self.__hash,),
                )
            )

            if len(rows) == 0:
                return None

            return np.array(rows[0])  # self._correct_laser(img, np.array(rows[0]))
