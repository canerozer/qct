from __future__ import annotations
import os
from typing import OrderedDict
from PIL import Image
import numpy as np
import random
from collections import OrderedDict

import torch
import torchvision.transforms as transforms


class ForeignObjectDataset(object):
    
    # def __init__(self, datafolder, datatype='train', transform=True, labels_dict={}, output_loc=False,
    #             config=None):
    def __init__(self, config, datatype='train', transform=True, labels_dict={}, 
                 output_name=False, output_loc=False,
                 shuffle=True, seed=1):
        print(config)
        datafolder = os.path.join(config.DATA.DATA_PATH, datatype)
        # datafolder = os.path.join(datafolder, datatype)
        self.datafolder = datafolder
        self.datatype = datatype
        if output_loc:
            labels_dict = dict(filter(lambda x: x[1] != '', labels_dict.items()))
        self.output_name = output_name
        self.labels_dict = labels_dict
        self.image_files_list = [s for s in sorted(os.listdir(datafolder)) if s in labels_dict.keys()]
        self.transform = transform
        self.annotations = [labels_dict[i] for i in self.image_files_list]
        self.output_loc = output_loc
        self.config = config


    def __getitem__(self, idx):
        # load images 
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.datafolder, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        
        boxes = []

        if self.datatype == 'train':
            annotation = self.labels_dict[img_name]

            if annotation == "":
                target = np.array(0)
            else:
                target = np.array(1)

            # convert everything into a torch.Tensor
            #boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            #labels = torch.ones((len(boxes),), dtype=torch.int64)

            #image_id = torch.tensor([idx])
            #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            #iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            #target = {}
            #target["boxes"] = boxes
            #target["labels"] = labels
            #target["image_id"] = image_id
            #target["area"] = area
            #target["iscrowd"] = iscrowd
            
            if self.transform is not None:
                img = self.transform(img)
                
            return img, target
        
        if self.datatype == 'dev':
            
            if self.labels_dict[img_name] == '':
                label = 0
            else:
                label = 1
            
            if self.transform is not None:
                img = self.transform(img)

            if self.output_loc:
                annotation = self.labels_dict[img_name]
                if type(annotation) == str:
                    annotation_list = annotation.split(';')
                    for anno in annotation_list:
                        x = []
                        y = []
                    
                        anno = anno[2:]
                        anno = anno.split(' ')
                        for i in range(len(anno)):
                            if i % 2 == 0:
                                x.append(float(anno[i]))
                            else:
                                y.append(float(anno[i]))
                            
                        xmin = min(x)/width * self.config.DATA.IMG_SIZE
                        xmax = max(x)/width * self.config.DATA.IMG_SIZE
                        ymin = min(y)/height * self.config.DATA.IMG_SIZE
                        ymax = max(y)/height * self.config.DATA.IMG_SIZE
                        boxes.append([xmin, ymin, xmax, ymax])

                return img, img_name, boxes

            if self.output_name:
                return img, label, img_name
            else:
                return img, label
        
        if self.datatype == 'test':
            
            if self.labels_dict[img_name] == '':
                label = 0
            else:
                label = 1
            
            if self.transform is not None:
                img = self.transform(img)

            if self.output_loc:
                annotation = self.labels_dict[img_name]
                if type(annotation) == str:
                    annotation_list = annotation.split(';')
                    for anno in annotation_list:
                        x = []
                        y = []
                    
                        anno = anno[2:]
                        anno = anno.split(' ')
                        for i in range(len(anno)):
                            if i % 2 == 0:
                                x.append(float(anno[i]))
                            else:
                                y.append(float(anno[i]))
                            
                        xmin = min(x)/width * self.config.DATA.IMG_SIZE
                        xmax = max(x)/width * self.config.DATA.IMG_SIZE
                        ymin = min(y)/height * self.config.DATA.IMG_SIZE
                        ymax = max(y)/height * self.config.DATA.IMG_SIZE
                        boxes.append([xmin, ymin, xmax, ymax])
                return img, img_name, boxes

            if self.output_name:
                return img, label, img_name
            else:
                return img, label

    def __len__(self):
        return len(self.image_files_list)


class OneToThreeDimension(object):
    """Convert BW to RGB Image"""

    def __call__(self, input):
        image = input
        image = np.expand_dims(image, axis=0)    
        image = np.repeat(image, 3, axis=0)
        return image


def retrieve_data_transforms(args, input_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tfs = []
    if args.STATE:
        tfs.append(transforms.Resize(input_size))
        if args.RANDOM_HOR_FLIP.STATE:
            tfs.append(transforms.RandomHorizontalFlip())
        if args.COLOR_JITTER.STATE:
            tfs.append(transforms.ColorJitter(args.COLOR_JITTER.BR,
                                              args.COLOR_JITTER.CON,
                                              args.COLOR_JITTER.SAT,
                                              args.COLOR_JITTER.HUE)
                      )
        if args.RANDOM_AFFINE.STATE:
            x = y = args.RANDOM_AFFINE.TR
            s_min = args.RANDOM_AFFINE.SC_MIN
            s_max = args.RANDOM_AFFINE.SC_MAX
            tfs.append(transforms.RandomAffine(args.RANDOM_AFFINE.DEG,
                                               (x, y),
                                               (s_min, s_max),
                                               args.RANDOM_AFFINE.SH)
                      )
        tfs.append(transforms.ToTensor())
        tfs.append(transforms.Normalize(mean=mean, std=std))
    else:
        tfs.append(transforms.Resize(input_size))
        tfs.append(transforms.ToTensor())
        tfs.append(transforms.Normalize(mean=mean, std=std))

    data_transforms = transforms.Compose(tfs)
    return data_transforms
