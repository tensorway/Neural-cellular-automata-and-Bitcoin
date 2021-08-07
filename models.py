import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque
import random
import copy
from utils import xsobel, ysobel

class Cell(nn.Module):
    def __init__(self, net_arch, last_activation = lambda x: x):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
        self.last_activation = last_activation
    def forward(self, x, stochastic=True, p_dropout=0.5):
        h = th.cat((x, xsobel(x), ysobel(x)), dim=1)
        h = h.transpose(1, -1)
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        h = h.transpose(1, -1)
        h = th.nn.functional.dropout2d(h, p=p_dropout, training=stochastic)
        return h