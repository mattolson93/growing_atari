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
import model
from time import sleep

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.stats import entropy
import gym
from atari_data import MultiEnvironment

from collections import deque
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict




from scipy.ndimage.filters import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--img_dir', type=str, default=None)

    parser.add_argument('--latent', type=int, default=16)
    parser.add_argument('--wae_latent', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--env', type=str, default='SpaceInvaders-v0')
    parser.add_argument('--enc_file', type=str, default=None)
    parser.add_argument('--gen_file', type=str, default=None)
    parser.add_argument('--Q', type=str, default="Q")
    parser.add_argument('--P', type=str, default="P")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent_file', type=str, default="")
    parser.add_argument('--frame_skip', type=int, default=8)
    parser.add_argument('--speed', type=float, default=.01)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--frames_to_cf', type=int, default=50)
    parser.add_argument('--cf_all_actions', type=int, default=0)
    parser.add_argument('--salient_intensity', type=int, default=333)
    parser.add_argument('--last_frame_diff', type=int, default=3)
    
    args = parser.parse_args()
    return args

def generate_counterfactual(z_n, desired_action, agent, P, speed = .01, MAX_ITERS = 5000):
    target_action = Variable(torch.LongTensor([desired_action])).cuda()

    z_n_value = z_n.data.cpu().numpy()
    #import pdb; pdb.set_trace()
    i = 0
    while True:
        z_n = Variable(torch.FloatTensor(z_n_value).cuda(), requires_grad=True)
        z_n = model.norm(z_n)
        #import pdb; pdb.set_trace()
        logit = agent.pi(P(z_n))
        logp_cf = F.log_softmax(logit, dim=1)
        
        #perform gradient descent
        loss = F.nll_loss(logp_cf, target_action)
        dc_dz = autograd.grad(loss, z_n, loss)[0]
        
        #update the latent agent
        z_n = z_n - (dc_dz * speed)
        
        #use the updated logit to generate a new frame
        z_n_value = z_n.data.cpu().numpy()
        
        #logging things
        action = logp_cf.max(1)[1].data

        
        # if we've reached the desired action, then break
        if (action[0] == desired_action) or i > MAX_ITERS:
            p_cf = logp_cf.exp().data.cpu().numpy()[0]
            max1 =  np.max(p_cf)
            max2 = np.partition(p_cf, -2)[-2]

            diff = max1 - max2
            epsilon = .05
            if diff >= epsilon or i > MAX_ITERS:
                print("selected a {} from pi of {}"
                    .format(action[0],logp_cf.exp().cpu().data[0].numpy()))
                print("Finished counterfactual after {} iterations".format(i))
                #print("Ending logits: {}".format(torch.exp(logp).data.cpu().numpy()))
                break
        del z_n
        i += 1

    #print("Finished generating counterfactual...")
    return z_n

#original: unchanged frame
#cf: the new counterfactual frame
#delta: change in a pixel required to make a notice
def get_changed_pixels(original, cf, delta=0.0001):
    diff = cf - original

    diff = np.abs(diff)
    diff[diff < delta] = 0
    diff = np.sum(diff, axis =2)
    max_diff = np.max(diff)
    #diff[diff > (max_diff/2)] = max_diff

    if max_diff > delta:
        diff = diff / max_diff 
    #added = np.sum(np.max(diff, delta), dim=2)
    #removed = np.sum(np.min(diff, delta), dim=2)

    #normalized_added = added / np.max(added)
    #normalized_removed = removed / np.min(removed)

    return diff



def saliency_on_atari_frame(saliency, atari, fudge_factor=330, channel=0, sigma=.75):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    #S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    S = saliency.astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() 
    S = fudge_factor*pmax * S / S.max()
    I = atari.astype('uint16')
    I[35:195,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

def generate_saliency(atari, original, cf, salient_intensity):
    
    d_pixels = get_changed_pixels(original, cf)
    return saliency_on_atari_frame(d_pixels, atari, salient_intensity)

FONT_FILE = '/nfs/eecs-fserv/share/cuda-9.0/cuda/jre/lib/fonts/LucidaSansRegular.ttf'
            
def immsave(file, pixels, text_to_add = "", size=200):
    np_img = imresize(pixels,size, interp = 'nearest')

    if text_to_add == "":
        imsave(file, np_img)
        return

    height_to_add = np.uint8(np_img.shape[0] / 8)
    width_to_add = np_img.shape[1]
    padding = np.zeros((height_to_add, width_to_add, 3))
    np_img = np.vstack([padding, np_img])

    img = Image.fromarray(np.uint8( np_img))
    d = ImageDraw.Draw(img)
    if os.path.isfile(FONT_FILE):
        fnt = ImageFont.truetype(FONT_FILE, np.uint8(height_to_add/3))
        d.text((0,0), text_to_add, font = fnt, fill=(255,255,255))
    else:
        d.text((0,0), text_to_add, fill=(255,255,255))

    img.save(file)



def printlog(s, img_dir, fname='log.txt', end='\n', mode='a'):
    print(s, end=end) 
    f=open(os.path.join(img_dir,fname),mode) 
    f.write(s+'\n') 
    f.close()

def get_low_entropy_states(agent, frames_to_cf, cur_envs, new_frame_bw):

    agent_state = Variable(torch.Tensor(new_frame_bw).cuda())
    agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)
    done = [False]
    i = 0
    entropies = []

    all_game_actions = defaultdict(int)    
    while done[0] == False:
        i+=1
        agent_state = torch.cat(list(agent_state_history), dim=1)#torch.cat(list(agent_state_history), dim=1)

        logit = agent.pi(agent(agent_state))
        p = F.softmax(logit, dim=1)

        actions = p.max(1)[1].data.cpu().numpy()
        _, new_frame_bw, _, done, _ = cur_envs.step(actions)
        all_game_actions[actions[0]] +=1

        agent_state_history.append(Variable(torch.Tensor(new_frame_bw).cuda()))
        
        probabilty_array = p.data.cpu().numpy()[0]
        cur_entropy = entropy(probabilty_array)
        entropies.append(cur_entropy)
        
    sorted_entropies = sorted(entropies)

    for i in range(len(p[0])):
        all_game_actions[i] +=0
    return sorted_entropies[min(frames_to_cf, len(entropies))-1], all_game_actions

def diff2d(first, second):
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            if first[i,j] != second[i,j]:
                print("i,j: {} {}    val: {} {}".format(i,j,first[i,j], second[i,j]))

def diff3d(first, second):
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            for k in range(first.shape[2]):
                if first[i][j][k] != second[i][j][k]:
                    print("i,j,k: {} {} {}    val: {} {}".format(i,j,k, first[i][j][k], second[i][j][k]))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_fancy_entropy(logits, frame):
    orig_p = softmax(logits)
    orig_entropy = entropy(orig_p)

    smaller_logits = np.delete(logits, np.argmax(logits))
    smaller_p = softmax(smaller_logits)
    smaller_entropy = entropy(smaller_p)
    printlog("{},{:.5f},{:.5f},{}".format(frame,orig_entropy.item(), smaller_entropy.item(), orig_p), "fancy_imgs_5000_5e-08_160.tar")

def calculate_rank(ddict_ranks, a):
    sorted_values = sorted(ddict_ranks.values())
    ranking = sorted_values.index(ddict_ranks[a])
    #import pdb; pdb.set_trace()
    return ranking

def run_game(encoder, generator,  agent, Q, P, envs, seed, img_dir, frames_to_cf= 15, speed = 1e-9, MAX_ITERS = 5000, cf_all_actions=1, salient_intensity = 333, last_frame_diff = 3):
    for model in [encoder, generator,Q,P]:
        model.eval()
        #for child in model.children():
        #    if type(child) in [nn.BatchNorm2d, nn.BatchNorm1d]:
        #        child.track_running_stats = False
        
    states = []
    values = []
    logps = []
    
    envs.seed(seed )  
    torch.manual_seed(seed)

    new_frame_rgb, new_frame_bw = envs.reset()
    action_description = envs.get_action_meanings()


    saves = envs.clone_full_state()
    min_entropy, ddict_ranks =  get_low_entropy_states(agent, frames_to_cf, envs, new_frame_bw)
    envs.restore_full_state(saves)
    torch.manual_seed(seed)
    #new_frame_rgb, new_frame_bw = envs.reset()

 
    state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()
    state_history = deque([state, state.clone(), state.clone(),state.clone()], maxlen=4)

    agent_state = Variable(torch.Tensor(new_frame_bw).cuda())
    agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)

    np.set_printoptions(precision=4)

    done = [False]
    i = 0
    cf_count = 0
    last_frame = -100
    while done[0] == False and cf_count < frames_to_cf :
        i+=1
        state = torch.cat(list(state_history), dim=1)
        agent_state = torch.cat(list(agent_state_history), dim=1)#torch.cat(list(agent_state_history), dim=1)

        z_a = agent(agent_state)
        value = agent.value(z_a)
        logits = agent.pi(z_a)
        #calculate_fancy_entropy(logits.cpu().data[0].numpy(), i)
        p = F.softmax(logits, dim=1)


        z = encoder(state)
        z_n = Q(z_a)
        reconstructed = generator(z, p)

        out_state = np.hstack(state[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())
        out_recon = np.hstack(reconstructed[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())
        output = np.vstack([out_state, out_recon]) * 255

        
        
        actions = F.softmax(agent.pi(P(Q(z_a))), dim=1).max(1)[1].data.cpu().numpy()

        atari_frame = envs.envs[0].render(mode='rgb_array')
        new_frame_rgb, new_frame_bw, _, done, _ = envs.step(actions)

        agent_state_history.append(Variable(torch.Tensor(new_frame_bw).cuda()))
        state_history.append(Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda())


        probabilty_array = p[0].data.cpu().numpy()
        cur_entropy = entropy(probabilty_array)
        

        if cur_entropy > min_entropy: continue
        if i - last_frame < last_frame_diff: continue
        last_frame = i
        #imsave(img_dir + "/state_bw_recon{}.png".format(i), reconstructed[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy()[-1].mean(2) * 255)
        #imsave(img_dir + "/state_bw_real{}.png".format(i), state[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy()[-1].mean(2) * 255)
        immsave(img_dir + "/state_rgb{}.png".format(i), output)

        cf_count += 1
        print("running value cf {} on frame {}".format(cf_count, i))
        files_to_save = []
        files_to_save1 = []
        files_to_save2 = []
        files_to_save3 = []
        files_to_save4 = []
        distances = []
        '''
        opposites = [1,0,5,4,3,2]
        if True:
            a = opposites[actions[0]]
        '''
        for a in range(envs.get_action_size()):
            if a == actions[0]: continue

            if a <= 0: continue
            '''if cf_all_actions != 1:
                if calculate_rank(ddict_ranks, a) < 4: continue

                #hack

                if actions[0] != 5 and a != 5:
                    continue'''

            cf_zn = generate_counterfactual(z_n, a, agent, P, speed, MAX_ITERS)

            p_cf      = F.softmax(agent.pi(P(cf_zn)), dim=1)

            cf = generator(z, p_cf)


            out_state = np.hstack(state[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())
            out_recon = np.hstack(reconstructed[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())
            out_cf    = np.hstack(cf[0].view(4,3,160,160).permute(0,2,3,1).cpu().data.numpy())

            output = np.vstack([out_state, out_recon, out_cf]) * 255

            #filename = '/cfframe_{:04d}_action{}{}.png'.format(i, a, action_description[a])
            #immsave(img_dir + filename, output)

            saliency_img = generate_saliency(atari_frame, out_recon[:,480:640,:], out_cf[:,480:640,:], salient_intensity) /255
            #filename = '/salientframe_{:04d}_action{}{}.png'.format(i, a, action_description[a])
            #immsave(img_dir + filename, saliency_img)

            #original input, saliency, CF
            demo_img = np.hstack([out_state[:,480:640,:], saliency_img[35:195,:], out_cf[:,480:640,:]]) * 255
            text_to_add = "Original action a:                                                                 Saliency, Time Step:                                                      Counterfactual action a': "
            text_to_add2 = "\n{} {: <9}                                                                                 {:04d}                                                                             {} {}".format(actions[0], action_description[actions[0]], i, a, action_description[a])

            cur_distance = np.linalg.norm(z_n[0].cpu().data.numpy()-cf_zn[0].cpu().data.numpy())
            file_details = '{:04d}_action{}r{}_cf{}r{}{}_{}.png'.format(i, action_description[actions[0]], calculate_rank(ddict_ranks, actions[0]), a,  calculate_rank(ddict_ranks, a), action_description[a], cur_distance)
            file = img_dir + '/demo' + file_details #/demo_{:04d}_action{}r{}_cf{}r{}{}.png'.format(i, actions[0], calculate_rank(ddict_ranks, actions[0]), a,  calculate_rank(ddict_ranks, a), action_description[a])
            immsave(file, demo_img, text_to_add + text_to_add2)
            files_to_save.append((file, demo_img, text_to_add + text_to_add2))
            files_to_save1.append((img_dir + '/output1_' + file_details, out_state[:,480:640,:]* 255))
            files_to_save2.append((img_dir + '/output2_' + file_details, saliency_img[35:195,:]* 255))
            files_to_save3.append((img_dir + '/output3_' + file_details, out_cf[:,480:640,:]* 255))
            files_to_save4.append((img_dir + '/bw_' + file_details, out_cf[:,480:640,:].mean(2)* 255))
            #save just salient
            '''immsave(img_dir + '/salient_' + file_details, saliency_img)
            immsave(img_dir + '/output1_' + file_details, out_state[:,480:640,:]* 255)
            immsave(img_dir + '/output2_' + file_details, saliency_img[35:195,:])
            immsave(img_dir + '/output3_' + file_details, out_cf[:,480:640,:]* 255)
            immsave(img_dir + '/bw_' + file_details, out_cf[:,480:640,:].mean(2)* 255)'''

            distances.append(cur_distance)

            #save BW version
            #immsave(img_dir + '/bw_' + file_details, out_cf[:,480:640,:].mean(2) * 255)
        '''immsave(*files_to_save[np.argmax(distances)]) #save only the best action
        immsave(*files_to_save1[np.argmax(distances)])
        immsave(*files_to_save2[np.argmax(distances)])
        immsave(*files_to_save3[np.argmax(distances)])
        imsave(*files_to_save4[np.argmax(distances)])'''

            

    #dont change state
    for model in [encoder, generator,Q,P]:
        model.train()
        #for child in model.children():
        #    if type(child) in [nn.BatchNorm2d, nn.BatchNorm1d]:
        #        child.track_running_stats = True


   

def main():
    #load models
    #load up an atari game
    #run (and save) every frame of the game
    #args = parse_args()

    args = parse_args()
    MAX_ITERS = args.iters
    speed = args.speed
    frames_to_cf = args.frames_to_cf
    seed = args.seed
    img_dir = args.img_dir
    if img_dir == None:
        img_dir = "imgs_{}_{}_{}_cfskip{}".format(args.iters, args.speed, args.agent_file[-7:], args.last_frame_diff)

    if args.enc_file == None or args.gen_file == None:
        print("Need to load models for the gen and enc")
        exit()

    if not os.path.isfile(args.agent_file):
        args.agent_file = args.env + ".model.80.tar"
        if not os.path.isfile(args.agent_file):
            print("bad agent_file")
            exit()


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

    if args.frame_skip % 2 ==0 and args.env == 'SpaceInvaders-v0':
        print("SpaceInvaders needs odd frameskip due to bullet alternations")
        args.frame_skip = args.frame_skip - 1

    #run every model on all frames (4*n frames))
    print('Loading model...')
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    #number of updates to discriminator for every update to generator
    envs = MultiEnvironment(args.env, 1, args.frame_skip)

    agent = model.Agent(envs.get_action_size()).cuda() #cuda is fine here cause we are just using it for perceptual loss and copying to discrim
    agent.load_state_dict(torch.load(args.agent_file))
    encoder = model.Encoder(args.latent).cuda()
    generator = model.Generator(args.latent, envs.get_action_size()).cuda()
    Q = model.Q_net(args.wae_latent).cuda()
    P = model.P_net(args.wae_latent).cuda()


    Q.load_state_dict(torch.load(args.Q, map_location=map_loc))
    P.load_state_dict(torch.load(args.P, map_location=map_loc))
        

    encoder.load_state_dict(torch.load(args.enc_file, map_location=map_loc))
    generator.load_state_dict(torch.load(args.gen_file, map_location=map_loc))

    encoder.eval()
    generator.eval()
    Q.eval()
    P.eval()
    os.makedirs(img_dir, exist_ok=True)
    '''def rm_file(filename):
        if os.path.exists(filename): os.remove(os.path.join(img_dir,filename))
    rm_file('log.txt')
    rm_file('probabilities.csv')
    rm_file('every_action.csv')
    

    #printlog('frame, entropy, max_less_avg, advantage_max_less_avg, l2_z, first_step, action, distance, iterations, Q[a], avg_step_size, l2_mag_cf, p_delta_start_action, p_delta_target_action, gold_delta_start_action, gold_delta_target_action', 'every_action.csv')
    printlog("frame, probabilty_array, entropy, Qs, average(Qs), advantage_max_less_avg", img_dir, 'probabilities.csv')
    '''
    print('finished loading models')
    
    run_game(encoder, generator, agent, Q, P, envs, seed, img_dir, frames_to_cf, speed, MAX_ITERS, args.cf_all_actions, args.salient_intensity, args.last_frame_diff)





if __name__ == '__main__':
    main()

