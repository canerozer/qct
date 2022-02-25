import os
import random
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class UKBiobankLVOTDataset(Dataset):
    def __init__(self, label_file, datatype=None, shuffle=True, transform=None, seed=0, debug=False,
                 output_loc=False):
        self.label_file = label_file
        self.datatype = datatype
        self.transform = transform
        self.seed = seed
        self.debug = debug
        self.output_loc = output_loc
        self.construct_labels()
        if not debug:
            self.read_images()
        self.shuffle(shuffle)

    def construct_labels(self):
        self.filenames = []
        self.labels =  []

        with open(self.label_file, 'r') as f:
            content = f.read().split("\n")[:-1]

        for line in content:
            fn, label = line.split(" ")
            self.filenames.append(fn)
            self.labels.append(int(label))

    def read_images(self):
        self.images = []
        for img_name in self.filenames:
            img = Image.open(img_name).convert("RGB")
            self.images.append(img)

    def shuffle(self, shuffle):
        if shuffle:
            len = self.__len__()
            idxs = list(range(len))
            random.Random(self.seed).shuffle(idxs)
            self.filenames = [self.filenames[idx] for idx in idxs]
            self.labels = [self.labels[idx] for idx in idxs]
            if not self.debug:
                self.images = [self.images[idx] for idx in idxs]

    def __getitem__(self, idx):
        if self.debug:
            img_name = self.filenames[idx]
            img = Image.open(img_name).convert("RGB")
        else:
            img = self.images[idx]
        target = np.array(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        # if self.datatype == "test":
        #     name = img_name.split("/")[-1]
        #     return img, target, name
        # else:
        return img, target

    def __len__(self):
        return len(self.filenames)
