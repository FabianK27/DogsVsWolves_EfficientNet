import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import random
from config import *
import matplotlib.pyplot as plt

class DogsVsWolvesDataset(Dataset):
    def __init__(self, root_dir):
        super(DogsVsWolvesDataset, self).__init__()
        self.root_dir = root_dir
        self.dog_files = os.listdir(os.path.join(root_dir, 'dogs'))
        self.wolf_files = os.listdir(os.path.join(root_dir, 'wolves'))

        self.dog_files = [(x, 'dog') for x in self.dog_files]
        self.wolf_files = [(x, 'wolf') for x in self.wolf_files]

        self.file_list = self.dog_files + self.wolf_files
        random.seed(123)
        random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name, image_label = self.file_list[idx]
        image_path = os.path.join(self.root_dir, 'dogs', image_name) if image_label == 'dog' \
            else os.path.join(self.root_dir, 'wolves', image_name)
        img = np.array(Image.open(image_path))

        augmentations = transform_dogs(image=img) if image_label == 'dog' \
            else transform_wolves(image=img)

        img = augmentations['image']

        return img, label_to_class[image_label]

