import os
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class HSICity2(Dataset):
    ignored_label = 255
    hsicity_label = [
            (0, (128, 64, 128)),  # road
            (1, (244, 35, 232)),  # sidewalk
            (2, (70, 70, 70)),  # building
            (3, (102, 102, 156)),  # wall
            (4, (190, 153, 153)),  # fence
            (5, (153, 153, 153)),  # pole
            (6, (250, 170, 30)),  # traffic light
            (7, (220, 220, 0)),  # traffic sign
            (8, (107, 142, 35)),  # vegetation
            (9, (152, 251, 152)),  # terrain
            (10, (70, 130, 180)),  # sky
            (11, (220, 20, 60)),  # person
            (12, (255, 0, 0)),  # rider
            (13, (0, 0, 142)),  # car
            (14, (0, 0, 70)),  # truck
            (15, (0, 60, 100)),  # bus
            (16, (0, 80, 100)),  # train
            (17, (0, 0, 230)),  # motorcycle
            (18, (119, 11, 32)),  # bicycle
        ]

    def __init__(self, root: str, use_hsi: bool = True, test_dir: str = None):
        super(HSICity2, self).__init__()
        self.root = root
        self.use_hsi = use_hsi
        self.test_dir = test_dir
        self.images = self._load()

    def __len__(self):
        return len(self.images)

    def _load(self) -> List[str]:
        images = []
        for img in os.listdir(self.root):
            if not img.endswith(".jpg"):
                continue
            name = img[3:-4]
            if self.test_dir is not None and ('rgb%s_gray.png' % name) not in os.listdir(self.test_dir):
                continue
            if name + '.hsd' not in os.listdir(self.root):
                continue
            images.append(name)
        return images

    @staticmethod
    def _read_hsd(filename: str) -> torch.Tensor:
        data = np.fromfile("%s" % filename, dtype=np.int32)
        height = data[0]
        width = data[1]
        SR = data[2]
        D = data[3]

        data = np.fromfile("%s" % filename, dtype=np.float32)
        a = 7
        average = data[a : a + SR]
        a = a + SR
        coeff = data[a : a + D * SR].reshape((D, SR))
        a = a + D * SR
        scoredata = data[a : a + height * width * D].reshape((height * width, D))

        temp = np.dot(scoredata, coeff)

        data = (temp + average).reshape((height, width, SR))

        return data

    def __getitem__(self, index):
        name = self.images[index]
        image = cv2.imread(
            os.path.join(self.root, "rgb" + name + ".jpg"), cv2.IMREAD_COLOR
        )
        label = cv2.imread(
            os.path.join(self.root, "rgb" + name + "_gray.png"), cv2.IMREAD_GRAYSCALE
        )

        h, w, _ = image.shape
        
        if self.use_hsi:
            hsi = self._read_hsd(os.path.join(self.root, name + ".hsd"))
            hsi = cv2.resize(hsi, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            hsi = hsi.transpose((2, 0, 1))

        image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2, 0, 1))
        image = image / 255.0
        
        if self.use_hsi:
            return image, label, hsi
        else:
            return image, label, image

    @staticmethod
    def to_stacks(
        image: np.ndarray,
        label: np.ndarray,
        hsi: np.ndarray,
        sz: int,
        ignored_label: int = 255,
    ):
        _, h, w = image.shape
        images = []
        labels = []
        hsis = []
        for i in range(sz, h, sz):
            for j in range(sz, w, sz):
                cube_label = label[i, j]
                if cube_label == ignored_label:
                    continue
                cube_image = image[:, i - sz : i, j - sz : j]
                cube_hsi = hsi[:, i - sz : i, j - sz : j]
                images.append(cube_image)
                labels.append(cube_label)
                hsis.append(cube_hsi.reshape((-1,)))
        return images, labels, hsis

    @classmethod
    def convert_label(cls, label: np.ndarray):
        temp = label.copy()
        for label_id, color in cls.hsicity_label:
            temp[temp == label_id] = color
        return temp
