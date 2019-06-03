import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch import autograd
from torch.autograd import Variable
import torch.multiprocessing as mp
import model

import signal
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#import logutil
import time
from math import log2
from atari_data import MultiEnvironment
import top_entropy_counterfactual

os.environ['OMP_NUM_THREADS'] = '1'

from scipy.misc import imsave
from scipy.misc import imresize

from collections import deque

#ts = logutil.TimeSeries('Atari Distentangled Auto-Encoder')


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--checkpoint_dir', type=str, default='')

parser.add_argument('--latent', type=int, default=128)
parser.add_argument('--env', type=str, default='SpaceInvaders-v0')
parser.add_argument('--agent_file', type=str, default="SpaceInvaders-v0.fskip7.160.tar")
parser.add_argument('--info', type=str, default="")
parser.add_argument('--epochs', help="measured in 100k frames", type=int, default=50)
parser.add_argument('--fskip', type=int, default=8)
parser.add_argument('--gpu', type=int, default=7)


args = parser.parse_args()

map_loc = {
        'cuda:0': 'cuda:'+str(args.gpu),
        'cuda:1': 'cuda:'+str(args.gpu),
        'cuda:2': 'cuda:'+str(args.gpu),
        'cuda:3': 'cuda:'+str(args.gpu),
        'cuda:4': 'cuda:'+str(args.gpu),
        'cuda:5': 'cuda:'+str(args.gpu),
        'cuda:7': 'cuda:'+str(args.gpu),
        'cuda:6': 'cuda:'+str(args.gpu),
        'cpu': 'cpu',
}

print('Initializing OpenAI environment...')
if args.fskip % 2 == 0 and args.env == 'SpaceInvaders-v0':
    print("SpaceInvaders needs odd frameskip due to bullet alternations")
    args.fskip = args.fskip -1


envs = MultiEnvironment(args.env, args.batch_size, args.fskip)
action_size = envs.get_action_size()

print('Building models...')
torch.cuda.set_device(args.gpu)
if not (os.path.isfile(args.agent_file) and  os.path.isfile(args.agent_file) and  os.path.isfile(args.agent_file)):
    print("need an agent file")
    exit()
    args.agent_file = args.env + ".model.80.tar"
agent = model.Agent(action_size).cuda()
agent.load_state_dict(torch.load(args.agent_file, map_location=map_loc))

Z_dim = args.latent

encoder = model.Encoder(Z_dim).cuda()
generator = model.Generator(Z_dim).cuda()

encoder.train()
generator.train()
    
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))

print('finished building model')

bs =  args.batch_size
TINY = 1e-9

def immsave(file, pixels, size=200):
    np_img = imresize(pixels,size, interp = 'nearest')
    imsave(file, np_img)

def model_step(input_images, target_images):
    optim_enc.zero_grad()
    optim_gen.zero_grad()

    gen_images = generator(encoder(input_images))

    loss = .5*torch.sum((gen_images - target_images)**2 ) / bs
    loss.backward()
    
    optim_enc.step()
    optim_gen.step()

    return loss.item()


input_size = 4
pred_size = 4
HUNDRED_K = 100000
def train(epoch):
    new_frame_rgb, new_frame_bw = envs.reset()

    state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()
    frames_to_save = input_size + pred_size
    state_history = deque([state.clone() for i in range(frames_to_save)], maxlen=frames_to_save)

    agent_state = Variable(torch.Tensor(new_frame_bw).cuda())
    agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)
    
    fs = 0
    for i in range(int( HUNDRED_K / bs)):
        #maintain current states
        state_history_list = list(state_history)

        historic_state = torch.cat(state_history_list[:input_size], dim=1)
        state = torch.cat(state_history_list[pred_size:], dim=1)

        agent_state = torch.cat(list(agent_state_history), dim=1)
        actions = F.softmax(agent.pi(agent(agent_state)), dim=1).max(1)[1].data.cpu().numpy()

        #updating the pred model
        ae_loss = model_step(historic_state, state)


        new_frame_rgb, new_frame_bw, _, done, _ = envs.step(actions)

        agent_state_history.append(Variable(torch.Tensor(new_frame_bw).cuda()))
        state_history.append(Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda())
        
        if np.sum(done) > 0:
            for j, d in enumerate(done):
                if d:
                    cur_state = state_history_list[-1][j].clone()
                    for item in state_history:
                        item[j] = cur_state.clone()
        
        if i % 20 == 0:
            print("Recon loss: {:.3f} ".format(ae_loss))
            if i % 300 == 0:
                fs = (i * args.batch_size) + (epoch * HUNDRED_K)
                #print("running running_average is: {}".format(running_average.item()))
                print("{} frames processed. {:.2f}% complete".format(fs , 100* (fs / (args.epochs * HUNDRED_K))))

    return ae_loss


def run_game(single_env, seed, img_dir, frame_count= 100):
    encoder.eval()
    generator.eval()
    
    single_env.seed(seed )  
    torch.manual_seed(seed)

    new_frame_rgb, new_frame_bw = single_env.reset()

    state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()
    frames_to_save = input_size + pred_size
    state_history = deque([state.clone() for i in range(frames_to_save)], maxlen=frames_to_save)

    agent_state = Variable(torch.Tensor(new_frame_bw).cuda())
    agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)
    
    np.set_printoptions(precision=4)

    done = [False]
    i = 0
    while done[0] == False and i < frame_count:

        #maintain current states
        state_history_list = list(state_history)

        historic_state = torch.cat(state_history_list[:input_size], dim=1)
        state = torch.cat(state_history_list[pred_size:], dim=1)

        agent_state = torch.cat(list(agent_state_history), dim=1)
        actions = F.softmax(agent.pi(agent(agent_state)), dim=1).max(1)[1].data.cpu().numpy()

        reconstructed = generator(encoder(historic_state))

        #save the prediction
        out_state = np.hstack(state[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())
        out_recon = np.hstack(reconstructed[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())
        output = np.vstack([out_state, out_recon]) * 255
        immsave(img_dir + "/state_rgb{}.png".format(i), output)
        i+=1
        

        new_frame_rgb, new_frame_bw, _, done, _ = single_env.step(actions)

        agent_state_history.append(Variable(torch.Tensor(new_frame_bw).cuda()))
        state_history.append(Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda())

    encoder.train()
    generator.train()

def save_models(epoch):
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen{}'.format(epoch)))
    torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'enc{}'.format(epoch)))
    

def make_graph(save_dir, X_plot, Y_train_loss):
    print("saving graphs")
    plt.grid(True)
    plt.plot(X_plot, Y_train_loss)
    plt.ylabel('MSE pred',fontweight='bold')
    plt.xlabel('epoch',fontweight='bold')
    plt.title('epoch vs predicted frames', fontsize=20)
    plt.savefig(save_dir +'/model_accuracy.png')
    plt.clf()

def main():
    print('creating directories')
    if args.checkpoint_dir == '':
        args.checkpoint_dir = "checkpoints_{}_latent{}_epochs{}".format(args.info, args.latent, args.epochs)
    
    os.makedirs(args.checkpoint_dir , exist_ok=True)

    losses = []

    for i in range(args.epochs):
        #img_dir = os.path.join(args.checkpoint_dir, "imgs"+str(i))
        #os.makedirs(img_dir, exist_ok=True)
        encoder.train()
        generator.train()
        cur_loss = train(i)
        losses.append(cur_loss)
        
        print("saving models to directory: {}". format(args.checkpoint_dir))
        save_models("")
        print("Evaluating the Autoencoder")
        test_env = MultiEnvironment(args.env, 1, args.fskip)
        run_game(test_env, 8, args.checkpoint_dir)
    
    make_graph(args.checkpoint_dir, [i for i in range(args.epochs)],losses)


if __name__ == '__main__':
    main()
