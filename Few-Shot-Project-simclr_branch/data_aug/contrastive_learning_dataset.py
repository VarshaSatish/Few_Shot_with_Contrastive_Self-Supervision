from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import os
import pickle
import numpy as np
import torch
from PIL import Image

class MiniImagenetDataset():
    def __init__(self, mode='train', root='/home/misc/RnD_project/mini-imagenet', transform=None, target_transform=None, n_views = 2):
        
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. Follow instructions to download mini-imagenet.')

        pickle_file = os.path.join(self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        f = open(pickle_file, 'rb')
        self.data = pickle.load(f)
        # print("Inside Dataloader",len(self.data['image_data'])) # = 38400
        self.x = [np.transpose(x, (2, 0, 1)) for x in self.data['image_data']]
        self.x = [torch.FloatTensor(x) for x in self.x]
        self.y = [-1 for _ in range(len(self.x))]
        class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = class_idx[class_name]
        self.s = 1
        self.n = len(self.x)
        self.color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(84,84)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([self.color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])


    def __getitem__(self, idx):
        x = self.x[idx]
        #print(x.shape)
        x = transforms.ToPILImage(mode='RGB')(x)
        s = 1
        data_transforms2 = ContrastiveLearningViewGenerator(self.data_transforms,2)
        images1 = data_transforms2(x)

        return images1, self.y[idx]

    def __len__(self):
        return self.n

    def _check_exists(self):
        return os.path.exists(self.root)

def index_classes(items):
    idx = {}
    for i in items:
        if (not i in idx):
            idx[i] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx
