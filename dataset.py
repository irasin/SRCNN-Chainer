from random import randint
from glob import glob
import numpy as np
import chainer
import cv2


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pattern, cropsize=128):
        self._paths = glob(pattern)
        self._cropsize = cropsize
        self._length = len(self._paths)

    def __len__(self):
        return self._length

    def get_example(self, i):
        while True:
            try:
                image = cv2.imread(self._paths[i], cv2.IMREAD_COLOR)
                assert image is not None
                h, w, _ = image.shape
                assert (w >= self._cropsize and h >= self._cropsize)
                break
            except AssertionError:
                i = (i + np.random.randint(1, self._length)) % self._length
        high_x = randint(0, w - self._cropsize)
        high_y = randint(0, h - self._cropsize)
        image_high = image[high_y:high_y+self._cropsize, high_x:high_x + self._cropsize]
        image_low = cv2.resize(image_high,
                               (int(self._cropsize / 2), int(self._cropsize / 2)),
                               interpolation=cv2.INTER_CUBIC)
        image_low = cv2.resize(image_low,
                               (self._cropsize, self._cropsize),
                               interpolation=cv2.INTER_CUBIC)

        return self.toChainer(image_high), self.toChainer(image_low)

    def toChainer(self, cv2_image):
        return cv2_image.astype(np.float32).transpose((2, 0, 1)) / 255.0
