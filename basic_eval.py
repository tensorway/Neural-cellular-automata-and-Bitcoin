#%%
import torch as th
import cv2
from utils import alive_conv, alive_pool, Buffer
from models import Cell
import numpy as np

hidden_size = 13
color_size = 3
state_size = hidden_size + color_size
downsampled_size = 64
device = th.device('cpu')
img_shape = (1, 3, 64, 64)
alive_thres = 0.1

#%%
buffer = Buffer(256)
cell = Cell([state_size*3, 128, state_size]).to(device)
cell.load_state_dict(th.load("cell_bitcoin_grow_from_one_5800.th"))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('basic.mp4', fourcc, 20.0, (512,512))

# %%
test_batch_size = 1
state = th.zeros(test_batch_size, state_size, img_shape[-2], img_shape[-1], device=device)
state[:, :, 32, 17] = th.tensor(th.ones(test_batch_size, state_size))
for step in range(200):
    past_alive = th.sigmoid( state[:, img_shape[1]].unsqueeze(1)*10 - 5)
    alive = th.sigmoid(alive_conv(past_alive)*10-5)
    with th.no_grad():
        alpha_channel = th.sigmoid(state[:, img_shape[1]].unsqueeze(1)*20-8)
        alive = (alive_pool(alpha_channel) > alive_thres).type(th.float32)
        dstate = cell(state)
        state = (dstate + state)
        state = state * alive

    toshow = state[0, :3].transpose(0, 2).transpose(0, 1).detach().cpu().numpy()
    aliveshow = state[0, 3:4].transpose(0, 2).transpose(0, 1).detach().cpu().numpy()
    cv2.imshow('alive', cv2.resize(aliveshow, (512, 512)))
    cv2.imshow('real thing', cv2.resize(toshow*255, (512, 512)))
    cv2.waitKey(1)
    out.write(cv2.resize(np.clip(toshow, 0, 1)*255, (512, 512)).astype(np.uint8))
out.release()