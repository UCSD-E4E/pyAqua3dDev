from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, file: Path) -> np.ndarray:
        raise NotImplementedError
