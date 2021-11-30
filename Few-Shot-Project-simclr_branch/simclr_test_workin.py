import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
from prototypical_loss import full_loss
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import torch.utils.data as data
import errno
from PIL import Image
import shutil
import pickle

from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules import Module
from mini_imagenet_dataset import MiniImagenetDataset
from pdb import set_trace as breakpoint

#device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

arch = 'resnet18'

if arch == 'resnet18':
  model = ResNetSimCLR('resnet18', out_dim=128)
elif arch == 'resnet50':
  model = ResNetSimCLR('resnet50', out_dim=128)

checkpoint = torch.load('checkpoint_0200.pth.tar', map_location='cpu')
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
#assert log.missing_keys == ['fc.weight', 'fc.bias']"
#print(log.missing_keys)

model.to(device)

test_dataset = MiniImagenetDataset(mode = 'test5')
print(len(test_dataset))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=300, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True)

#print(len(test_loader))

top1_accuracy = 0
top5_accuracy = 0
ii = 0

avg_acc=[]

for counter, (x_batch, y_batch) in enumerate(test_loader):
    #print('x',len(x_batch))
    #print('y', len(y_batch))
    x_batch = torch.FloatTensor(x_batch)
    y_batch = y_batch.type(torch.FloatTensor)

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    l, acc = full_loss(logits, target=y_batch, n_support=1+++++++++, inner_loop=True)
    acc = acc.squeeze().detach().cpu().numpy()
    #breakpoint()
    #print(acc)
    avg_acc.append(acc)
avg_acc = np.mean(avg_acc)
print('Test Acc: {}'.format(avg_acc))
