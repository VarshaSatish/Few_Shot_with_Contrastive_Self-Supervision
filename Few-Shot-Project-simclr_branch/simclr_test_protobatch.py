import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
from prototypical_loss import full_loss
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
from prototypical_batch_sampler import PrototypicalBatchSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

classes_per_it_val = 5
num_support_val = 5
num_query_val = 5
iterations = 50
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

test_sampler = PrototypicalBatchSampler(labels=test_dataset.y,
                                            classes_per_it=classes_per_it_val,
                                            num_samples=num_support_val + num_query_val,
                                            iterations=iterations)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_sampler=test_sampler)


test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=300, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True)

print(len(test_loader))

top1_accuracy = 0
top5_accuracy = 0
ii = 0

avg_acc=[]

for epoch in range(10):
    test_iter = iter(test_dataloader)
    for batch in test_iter:
        x, y = batch
        x, y = Variable(x), Variable(y)
        
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        l, acc = full_loss(logits, target=y, n_support=5, inner_loop=True)
        acc = acc.squeeze().detach().cpu().numpy()
        avg_acc.append(acc)
avg_acc = np.mean(avg_acc)
print('Test Acc: {}'.format(avg_acc))