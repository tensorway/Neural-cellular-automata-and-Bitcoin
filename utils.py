#%%
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

hidden_size = 13
color_size = 3
device = th.device('cpu')
state_size = hidden_size + color_size
xsobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
ysobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
alivel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
depth = state_size
xsobel_kernel = th.tensor(xsobel, dtype=th.float32, device=device).unsqueeze(0).expand(depth, 1, 3, 3)
ysobel_kernel = th.tensor(ysobel, dtype=th.float32, device=device).unsqueeze(0).expand(depth, 1, 3, 3)
alive_kernel  = th.tensor(alivel, dtype=th.float32, device=device).unsqueeze(0).expand(1, 1, 3, 3)
xsobel = lambda x: F.conv2d(x, xsobel_kernel, stride=1, padding=1, groups=x.size(1))
ysobel = lambda x: F.conv2d(x, ysobel_kernel, stride=1, padding=1, groups=x.size(1))
alive_conv = lambda x: F.conv2d(x, alive_kernel, stride=1, padding=1, groups=x.size(1))
alive_pool = lambda x: F.max_pool2d(x, 3, padding=1, stride=1)


class Buffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.deque = deque()
    def append(self, item):
        if len(self.deque) > self.maxlen:
            self.deque.pop()
        self.deque.appendleft(item)
    def sample(self, n):
        return random.sample(self.deque, n)
    def __len__(self):
        return len(self.deque)