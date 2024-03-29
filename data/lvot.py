import os
import random
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class LVOTDataset(Dataset):
    def __init__(self, label_file, datatype=None, shuffle=True, transform=None, seed=0, debug=False,
                 output_loc=False, output_name=False, config=None, segm_folder=None):
        self.label_file = label_file
        self.datatype = datatype
        self.transform = transform
        self.seed = seed
        self.debug = debug
        self.output_loc = output_loc
        self.construct_labels()
        if not debug:
            self.read_images()
        self.output_loc = output_loc
        if output_loc:
            self.config = config
            self.segm_folder = config.SEGM_FOLDER
            # self.process_segmentation_maps()
        self.output_name = output_name
        self.shuffle(shuffle)

    def process_segmentation_maps(self):
        raise NotImplementedError
        # self.segm_maps = []
        # for fn in self.filenames:
        #     segm_map_path = os.path.join(self.segm_folder, fn)
        #     segm_map = Image.open(segm_map_path).convert('1')
        #     print(segm_map.min(), segm_map.max())




    def construct_labels(self):
        self.filenames = []
        self.fullpaths = []
        self.labels =  []

        with open(self.label_file, 'r') as f:
            content = f.read().split("\n")[:-1]

        for line in content:
            fullpath, label = line.split(" ")
            fn = fullpath.split('/')[-1]
            self.filenames.append(fn)
            self.fullpaths.append(fullpath)
            self.labels.append(int(label))

    def read_images(self):
        self.images = []
        for img_name in self.fullpaths:
            img = Image.open(img_name).convert("RGB")
            self.images.append(img)

    def shuffle(self, shuffle):
        if shuffle:
            len = self.__len__()
            idxs = list(range(len))
            random.Random(self.seed).shuffle(idxs)
            self.filenames = [self.filenames[idx] for idx in idxs]
            self.fullpaths = [self.fullpaths[idx] for idx in idxs]
            self.labels = [self.labels[idx] for idx in idxs]
            if not self.debug:
                self.images = [self.images[idx] for idx in idxs]

    def __getitem__(self, idx):
        img_name = self.fullpaths[idx]
        if self.debug:
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
        if self.output_loc:
            return img, target, img_name, None
        
        if self.output_name:
            return img, target, img_name

        return img, target

    def __len__(self):
        return len(self.fullpaths)
