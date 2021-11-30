# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import errno
import os
from PIL import Image
import torch
import shutil

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test5.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='/home/misc/RnD_project/omniglot', transform=None, target_transform=None, download=False):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes, mode = mode)

        self.idx_classes = index_classes(self.all_items, mode = mode)

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])

        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join('/', [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

   
def find_items(root_dir, classes, mode):
    retour = []
    rots = ['/rot000', '/rot090', '/rot180', '/rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split('/')
            lr = len(r)
            label = r[lr - 2] + "/" + r[lr - 1]
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    print("== {0} Dataset: Found {1} items ".format(mode,len(retour)))
    return retour


def index_classes(items, mode):
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    
    print("== {0} Dataset: Found {1} classes".format(mode,len(idx)))
    return idx


def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().splitlines()
    return classes


def load_img(path, idx):
    path, rot = path.split('/rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)
      
    return x
