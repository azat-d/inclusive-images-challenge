import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


class LabeledImages(Dataset):
    def __init__(self, list_path, images_root, transform=None):
        self._transform = transform
        self._images_root = images_root

        self._content = []
        self._labels_list = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')

                self._content.append(parts[0])
                labels = [int(label) for label in parts[1:]]
                self._labels_list.append(labels)

    def __len__(self):
        return len(self._content)

    def __getitem__(self, idx):
        img_path = os.path.join(self._images_root, self._content[idx])
        labels = np.array(self._labels_list[idx], dtype=np.int64)

        img = pil_loader(img_path)
        if self._transform is not None:
            img = self._transform(img)

        return {'image': img, 'labels': labels}