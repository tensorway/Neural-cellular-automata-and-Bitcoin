#%%
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
import time
import random
import copy
from models import Cell
from utils import Buffer, alive_pool


hidden_size = 13
color_size = 3
state_size = hidden_size + color_size
downsampled_size = 64
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
device = th.device('cpu')
#%%
# load the image for which cellular automata
#  will be trained for
nimg = cv2.imread('bitcoin.png')
nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
img = th.tensor(nimg, dtype=th.float32).transpose(0, 2).transpose(1, 2)
img = img[:, 10:185, 40:210]
img = tv.transforms.ToPILImage()(img)
img = tv.transforms.Resize((downsampled_size, downsampled_size))(img)
img = tv.transforms.ToTensor()(img)
img = img.unsqueeze(0).to(device)
plt.imshow(img[0, 0].cpu().numpy())
img.shape, img.shape[0]

#%%
# intializing everything
buffer = Buffer(256)
cell = Cell([state_size*3, 128, state_size]).to(device)
opt = th.optim.Adam(cell.parameters(), lr=1e-3)
cell.load_state_dict(th.load("cell_bitcoin_grow_from_one_5800.th"))

#%%
nepoch = 2000
nsteps = 500
loss_after_step = 30
batch_size = 2
alive_thres = 0.1
tlastprint = time.time()
alive_mask = 1-(img == img[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)).type(th.float32)[:, 0, :, :]


#%%
## basic train loop no regeneration
# regeneration could be tried at test time but
# the net with this loop will not be trained for
# that
writer = SummaryWriter()
for ep in range(0, nepoch):
    loss, loss_alive, loss_rgb = 0, 0, 0
    state = th.zeros(batch_size, state_size, img.shape[-2], img.shape[-1], device=device)
    state[:, :, 32, 17] = th.cat((th.zeros(batch_size, 3), th.ones(batch_size, state_size-3)), dim=1)
    nsteps_now = ep/nepoch*nsteps + 5
    loss_after_step_now = ep/nepoch*loss_after_step + 3
    for step in range(int(nsteps_now)):
        # past_alive = th.sigmoid( state[:, img.shape[1]].unsqueeze(1)*40 - 5)
        # alive = th.sigmoid(alive_conv(past_alive)*40-5)
        alpha_channel = th.sigmoid(state[:, img.shape[1]].unsqueeze(1)*20-8)
        alive = (alive_pool(alpha_channel) > alive_thres).type(th.float32)
        dstate = cell(state)
        state = (dstate + state)

        if step > loss_after_step_now:
            preds_rgb = state[:, :img.shape[1]]
            preds_alive = th.sigmoid(state[:, img.shape[1]]*20-8)
            loss_alive += ((preds_alive-alive_mask)**2).mean()
            loss_rgb += ((img-preds_rgb)**2).mean()
        state = state * alive
        if random.random() < 0.08: ##prob that nothing in 50 steps is added is 0.21
            buffer.append(state[-1].unsqueeze(0).detach().cpu())

    loss_alive /= (nsteps_now-loss_after_step_now)
    loss_rgb /= (nsteps_now-loss_after_step_now)
    loss = 5*loss_rgb + loss_alive
    opt.zero_grad()
    loss.backward()
    opt.step()
    if time.time() - tlastprint > 5:
        print(ep, loss.item(), nsteps_now, loss_after_step_now, len(buffer))
        tlastprint = time.time()
    if ep%200 == 0:
        th.save(cell.state_dict(), "cell_bitcoin_grow_from_one_"+str(ep)+".th")


    writer.add_scalar('loss/all', loss, ep)
    writer.add_scalar('loss/alive', loss_alive, ep)
    writer.add_scalar('loss/rgb', loss_rgb, ep)
    writer.add_image('final_image', state[0, :3], ep)
    writer.add_image('alive', alive[0].type(th.float32), ep)


#%%
# train loop that incorporates regeneration
# it traines to mimic the image while randomly
# killing cells
nbasic = 1
batch_size = 4
writer = SummaryWriter()

for ep in range(0, nepoch):
    loss, loss_alive, loss_rgb = 0, 0, 0

    one_cell_state = th.zeros(nbasic, state_size, img.shape[-2], img.shape[-1], device=device)
    one_cell_state[:, :, 32, 17] = th.cat((th.zeros(nbasic, 3), th.ones(nbasic, state_size-3)), dim=1)
    samples_state = th.cat(buffer.sample(batch_size-nbasic))
    state = th.cat((one_cell_state, samples_state))

    nsteps_now = ep/nepoch*nsteps + 50
    loss_after_step_now = 30#min(1, ep/nepoch)*loss_after_step + 3

    for step in range(int(nsteps_now)):
        alpha_channel = th.sigmoid(state[:, img.shape[1]].unsqueeze(1)*20-8)
        alive = (alive_pool(alpha_channel) > alive_thres).type(th.float32)
        dstate = cell(state)
        state = dstate + state

        if step > loss_after_step_now:
            preds_rgb = state[:, :img.shape[1]]
            preds_alive = th.sigmoid(state[:, img.shape[1]]*20-8)
            loss_alive += ((preds_alive-alive_mask)**2).mean()
            loss_rgb += ((img-preds_rgb)**2).mean()
        state = state * alive
        if random.random() < 0.03: ##prob that nothing in 50 steps is added is 0.21
            buffer.append(state[-1].unsqueeze(0).detach())

        for curri in range(nbasic, batch_size):
            if random.random() < 0.045:
                i, j = random.randint(0, img.shape[2]), random.randint(0, img.shape[3])
                h, w = random.randint(0, img.shape[2]//3), random.randint(0, img.shape[3]//3)
                h, w = min(img.shape[2]-i, h), min(img.shape[3]-j, w)
                zeroed = copy.deepcopy(state[curri].detach())
                zeroed[:, i:i+h, j:j+w] = th.zeros(zeroed.shape[0], h, w)
                if th.abs(zeroed[3].sum()) > 0.1:
                    ## Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
                    state[curri, :, i:i+h, j:j+w] = th.zeros(zeroed.shape[0], h, w)
        


    loss_alive /= (nsteps_now-loss_after_step_now)
    loss_rgb /= (nsteps_now-loss_after_step_now)
    loss = 5*loss_rgb + loss_alive
    opt.zero_grad() 
    loss.backward()
    opt.step()
    # print(alive.shape)
    if time.time() - tlastprint > 5:
        print(ep, loss.item(), nsteps_now, loss_after_step_now, len(buffer))
        tlastprint = time.time()
    if ep%200 == 0:
        th.save(cell.state_dict(), "cell_bitcoin_grow_from_one_reger_"+str(ep)+".th")


    writer.add_scalar('loss/all', loss, ep)
    writer.add_scalar('loss/alive', loss_alive, ep)
    writer.add_scalar('loss/rgb', loss_rgb, ep)
    writer.add_image('final_image', state[0, :3], ep)
    writer.add_image('alive', alive[0].type(th.float32), ep)

# %%
state = th.randn(1, state_size, img.shape[-2], img.shape[-1])
state[0, :3] = img
state = cell(state)
plt.imshow(state[0].sum(dim=0).detach().numpy())