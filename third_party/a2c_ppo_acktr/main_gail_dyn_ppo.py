#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov and (c) 2020 Google LLC
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from cProfile import label
import os
import time
from collections import deque
# pip install gym
import gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from third_party.a2c_ppo_acktr import algo, utils
from third_party.a2c_ppo_acktr.algo import gail
from third_party.a2c_ppo_acktr.arguments import get_args
from third_party.a2c_ppo_acktr.envs import make_vec_envs
from third_party.a2c_ppo_acktr.model import Policy
from third_party.a2c_ppo_acktr.model_split import SplitPolicy
from third_party.a2c_ppo_acktr.storage import RolloutStorage
from load import Load 
from equations import System
import matplotlib.pyplot as plt
import pdb

# from my_pybullet_envs import utils as gan_utils

import logging
import sys
from scipy.signal import savgol_filter

TRAJECTORY_LOAD_PATH1="./data/data/"
TRAJECTORY_LOAD_PATH2= "./data/parameters"
# TRAJECTORY_LENGTH = 30
TRAJECTORIES_NUM = 500 # MAX 90 
BATCH_SIZE = 1
# HIDDEN_SIZE = 16
# EPOCHS = 1000
SEED = np.random.random()
NUM_MINI_BATCH = 1
ENTROPY_COEF = 0
LR = 5*1e-8
GAIL_DIS_HDIM = 64
GAIL_TRAJ_NUM = 10
GAIL_DOWNSAMPLE_FREQUENCY = 1
GAIL_EPOCH = 10
NUM_STEPS = 2
NUM_PROCESSES = 1
NUM_ENV_STEPS = 1000
NUM_EPISODES = 50
CUDA = 1
TEST_NUM_STEPS = 2
TRAJ_LENGTH = 20

sys.path.append("third_party")
np.set_printoptions(precision=2, suppress=None, threshold=sys.maxsize)

def main():
    args, extra_dict = get_args()
    load = Load()
    xcommau = load.get_data(TRAJECTORY_LOAD_PATH1, TRAJECTORIES_NUM) # Data in the form of SAS, features = 9
    paramsdata = load.get_data(TRAJECTORY_LOAD_PATH2, TRAJECTORIES_NUM)
    trajlength = min(xcommau.shape[0], paramsdata.shape[0])
    xcommau = np.array(xcommau[:trajlength])
    paramsdata = np.array(paramsdata[:trajlength])
    print("PRINTING SHAPES OF THE TWO DATA INPUTS - ")
    print(np.shape(xcommau))
    print(np.shape(paramsdata))
    sasdata = np.zeros((np.shape(xcommau)[0], np.shape(xcommau)[1] + 2))
    for i in range(np.shape(xcommau)[0]-1):
        sasdata[i, :] = np.concatenate((xcommau[i], np.array([xcommau[i+1,0]]), np.array([xcommau[i+1,1]])))
    
    sasdata = sasdata[:-1]
    print(np.shape(sasdata))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if CUDA and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    # save_path = os.path.join(args.save_dir, args.algo)

    torch.set_num_threads(4)
    device = torch.device("cuda:0" if CUDA else "cpu")
    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    actor_critic = SplitPolicy(3, 2)
    actor_critic.to(device)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            NUM_MINI_BATCH,
            args.value_loss_coef,
            ENTROPY_COEF,
            lr=LR,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    
    expertdataset = TensorDataset(Tensor(np.array(sasdata)))
    drop_last = len(expertdataset) > BATCH_SIZE
    gail_train_loader = DataLoader(expertdataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    expertparamsdataset = TensorDataset(Tensor(np.array(paramsdata)))
    params_data_loader = DataLoader(expertparamsdataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    
    s_dim = 2
    a_dim = 1
    s_idx = np.array([0])
    a_idx = np.array([0])

    # # info_length = len(s_idx) * s_dim + len(a_idx) * a_dim + s_dim       # last term s_t+1   # 2 + 1 + 2 (S + a + S)
    # Defining the Discriminator 
    discr = gail.Discriminator(
        5, GAIL_DIS_HDIM,
        device)

    gail_tar_length = np.shape(sasdata)[0] * 1.0 / GAIL_TRAJ_NUM * GAIL_DOWNSAMPLE_FREQUENCY
    # print("GAIL TAR LENGTH = " + str( np.shape(sasdata)[0]))

    system = System()

    # obs = Tensor(xcommau[0,0:3])
    obs = Tensor(system.generate_initial_obs())
    print("First Obs = " + str(obs))

    # Creating empty object to store state,action values#################
    rollouts = RolloutStorage(NUM_STEPS, NUM_PROCESSES,
                              3,
                              actor_critic.recurrent_hidden_state_size, 5)
    # reset does not have info dict, but is okay,
    # and keep rollouts.obs_feat[0] 0, will not be used, insert from 1 (for backward compact)
    print("----------------")
    print(np.shape(rollouts.obs[0][0]))
    print(np.shape(obs))
    print("----------------")
    rollouts.obs[0][0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10000)
    gail_rewards = deque(maxlen=10)  # this is just a moving average filter
    total_num_episodes = 0
    j = 0
    max_num_episodes = NUM_EPISODES if NUM_EPISODES else np.infty

    start = time.time()
    num_updates = int(
        NUM_ENV_STEPS) // NUM_STEPS // NUM_PROCESSES

    from third_party.a2c_ppo_acktr.baselines.common.running_mean_std import RunningMeanStd
    ret_rms = RunningMeanStd(shape=())
    # errorcx = []
    mainerrorarr = []
    valuelossarr = []
    actionlossarr = []
    avgparamerror1 = []
    avgparamerror2 = []
    gailloss = []
    gaillossp = []
    gaillosse = []
    genloss = []
    merrorarr = []
    verrorarr = []
    fig, axs = plt.subplots(2 , 4)
    # fig2, axs2 = plt.subplots(2 , 1)

    while j < num_updates and total_num_episodes < max_num_episodes:

        obs = Tensor(system.generate_initial_obs())
        print("First Obs = " + str(obs))
        rollouts.obs[0][0].copy_(obs)
        

        
        # Learning rate decay for the main loop
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else LR)
        errorcxexpert = []
        paramsarr = []
        action_logstd_arr = []
        # obs = Tensor(system.generate_initial_obs())
        # rollouts.obs[0][0].copy_(obs)
        # rollouts.to(device) 
        # INITALLY COLLECTING TRAJECTORIES FROM CURRENT POLICY FROM THE SIMULATOR 
        for step in range(NUM_STEPS):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, action_logstd = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            action_logstd_arr.append([np.array(action_logstd[0][0]), np.array(action_logstd[0][1])])
            done1 = 1
            done2 = 1

            next_state, next_obs, errorcx = system.step(rollouts.obs[step][0][0:2], action)
            reward = Tensor(1) #-((leadernextstate[0] - followernextstate[0] - 20) + (leadernextstate[1] - followernextstate[1]))
            next_state = Tensor(next_state.cuda())
            next_obs = Tensor(next_obs).cuda()

            paramsarr.append(action.cpu().numpy())

            sas_feat = torch.cat((rollouts.obs[step][0], next_state[:,0])) #Tensor(np.concatenate(np.array(obs),np.array(next_state)))
            if (next_state[0] < 30 and next_state[0] > -30) and np.abs(next_obs[2].cpu().numpy()) < 50:
                done1 = 0
            if (next_state[1] < 30 and next_state[1] > -30) and np.abs(next_obs[2].cpu().numpy()) < 50:
                done2 = 0

            masks = Tensor(
                [done1, done2])
            bad_masks = Tensor(
                [1.0, 1.0])
            rollouts.insert(next_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, Tensor(sas_feat))

        # mainerrorarr.append(np.mean(errorcxexpert))
        # avgparamerror.append(np.mean(np.abs(paramsarr)))
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1][0], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        gail_loss, gail_loss_e, gail_loss_p = None, None, None
        gail_epoch = GAIL_EPOCH
        
        # x1arr = []
        # x2arr = []

        for _ in range(gail_epoch):
            gail_loss, gail_loss_e, gail_loss_p, gen_loss, merror, verror = discr.update_gail_dyn(gail_train_loader, rollouts)
            # x1arr.append(paramserror1)
            # x2arr.append(paramserror2)
            merrorarr.append(float(merror))
            verrorarr.append(float(verror))
            gaillosse.append(gail_loss_e)
            gaillossp.append(gail_loss_p)
            gailloss.append(gail_loss)
            genloss.append(gen_loss)
        
        
        # gailloss.append(gail_loss)
        # avgparamerror1.append(np.mean(x1arr))
        # avgparamerror2.append(np.mean(x2arr))

        num_of_dones = (1.0 - rollouts.masks).sum().cpu().numpy() \
            + NUM_PROCESSES / 2
        num_of_expert_dones = (NUM_STEPS * NUM_PROCESSES) / gail_tar_length

        d_sa = 1 - num_of_dones / (num_of_dones + num_of_expert_dones)

        if args.no_alive_bonus:
            r_sa = 0
        else:
            r_sa = np.log(d_sa) - np.log(1 - d_sa)  # d->1, r->inf
        # gailloss.append(r_sa)
        experttrajnorm = np.linalg.norm(xcommau[:,0:2], 2)
        # # print("L2 NORM OF THE EXPERT TRAJECTORY = " + str(experttrajnorm))

        # # print("THE DISCRIMINATOR PROBABILITY IS HERE : ")
        # discprob = discr.trunk(rollouts.obs)
        # print(discprob)
        for step in range(NUM_STEPS):
            rollouts.rewards[step], returns = \
                discr.predict_reward_combined(
                    rollouts.obs_feat[step + 1], args.gamma,
                    rollouts.masks[step], offset=-r_sa
                )

            ret_rms.update(returns.view(-1).cpu().numpy())
            rews = rollouts.rewards[step].view(-1).cpu().numpy()
            rews = np.clip(rews / np.sqrt(ret_rms.var + 1e-7),
                           -20.0, 20.0)
            rollouts.rewards[step] = Tensor(rews).view(-1, 2)
            # print("TORCH REWARDS = " + str(torch.mean(returns).cpu().data))
            currenttrajnorm = np.linalg.norm(rollouts.obs[:,0,0:2].cpu().numpy(), 2) 
            # print("ROLLOUTS.OBS  SHAPE = " + str(np.shape(rollouts.obs)))            
            rewardscale = 0.01
            # if currenttrajnorm < experttrajnorm:
            #     trajrew = rewardscale*()
            # else: 
            #     trajrew = -5*rewardscale
            # print("TRAJ REWARD = " + str(rewardscale*(experttrajnorm-currenttrajnorm)))
            finalrew = torch.mean(returns).cpu().data# + rewardscale*(experttrajnorm-currenttrajnorm)
            # print(finalrew)
            # rollouts.rewards[step] += rewardscale*(experttrajnorm-currenttrajnorm)
            gail_rewards.append(finalrew)
        
        # print("SHAPE OF ROLLOUTS.REWARDS = " + str(np.shape(rollouts.rewards)))

        # rollouts.rewards 

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, not args.no_proper_time_limits)
        
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        
        # rollouts.after_update()
        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))

        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + "_" + str(j) + ".pt"))

        #     if args.gail:
        #         torch.save(discr, os.path.join(save_path, args.env_name + "_D.pt"))
        #         torch.save(discr, os.path.join(save_path, args.env_name + "_" + str(j) + "_D.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * NUM_PROCESSES * NUM_STEPS
            end = time.time()
            root_logger.info(
                ("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes:" +
                 " mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, " +
                 "dist en {}, l_pi {}, l_vf {}, recent_gail_r {}," +
                 "loss_gail {}, loss_gail_e {}, loss_gail_p {}\n").format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss, np.mean(gail_rewards),
                    gail_loss, gail_loss_e, gail_loss_p
                )
            )
            # actor_critic.dist.logstd._bias,

        total_num_episodes += len(episode_rewards)
        episode_rewards.clear()
        j += 1
        # print("{}th CYCLE DONE".format(j))
        print("VALUE LOSS = " + str(value_loss))
        print("ACTION LOSS = " + str(action_loss))
        valuelossarr.append(value_loss)
        actionlossarr.append(action_loss)

        position1 = rollouts.obs[:,0,0]
        position2 = rollouts.obs[:,0,1]

        # print((position1).shape)
        l_to_show = position1.shape

        # axs[0].set_xlim([-50, 50])
        plt.ylim([-50, 50])
        # axs[1].set_xlim([-5, 5])
        # axs.set_ylim([-50, 50])
        print(action_logstd_arr)
        axs[0][0].cla()
        axs[0][1].cla()
        axs[1][0].cla()
        axs[1][1].cla()
        axs[1][2].cla()
        axs[1][0].set_title("X1 (Sim. vs Exp.")
        axs[1][1].set_title("X2 (Sim. vs Exp.")
        axs[1][2].set_title("REl. Error in Cx")
        axs[0][0].plot(gaillossp,color='blue', label='GAIL_LOSS_P')
        axs[0][0].plot(gaillosse, color='red', label='GAIL_LOSS_E')
        axs[0][0].plot(gailloss, color='green', label='GAIL_LOSS')
        axs[0][0].plot(genloss, color='black', label="GEN_LOSS")
        axs[1][0].plot(position1.cpu().numpy(), color='red', label='Predicted')
        # axs[1][2].plot(np.array(np.array(action_logstd_arr)[:,0]))
        # axs[1][2].plot(np.array(np.array(action_logstd_arr)[:,1]))
        for i in range(19):
            axs[1][0].plot(xcommau[20*i:20*(i+1),0], color='blue')
        axs[1][1].plot(position2.cpu().numpy(), color='red')
        for i in range(19):
            axs[1][1].plot(xcommau[20*i:20*(i+1),1], color='blue')
        # # axs[0].plot(valuelossarr)
        # axs[1].plot(gailloss)
        axs[0][0].set_title("GAIL LOSS")
        # axs[1].set_title("Plot Position 2")
        axs[0][0].legend()
        # axs[1].plot(position2.cpu().numpy())
        # print(np.shape(paramsarr))
        # axs[1].plot(np.array(paramsarr)[:,0,0], np.array(paramsarr)[:,0,1], color='red', label='Predicted')
        # axs[1].plot(paramsdata[:,0], paramsdata[:,1], color='blue', label='Expert')
        # axs[1].set_title("Params")
        # axs[1].legend()
        # axs[2].plot(delta_predition)
        # axs[2].set_title("Delta Prediction")
        plt.pause(0.05)
        
    # fig, axs2 = plt.subplots(2 , 1)
    # # axs2[0].plot(gail_rewards)
    # # axs2[1].plot(avgparamerror2)
    # axs2[0].plot(gaillossp,color='blue', label='GAIL_LOSS_P')
    # axs2[0].plot(gaillosse, color='red', label='GAIL_LOSS_E')
    # axs2[0].plot(gailloss, color='green', label='GAIL_LOSS')
    # axs2[0].plot(genloss, color='black', label="GEN_LOSS")
        
    # plt.show()

    print("-----------------------------------------")
    print("TESTING TRAINED DYNAMICS MODEL COMMENCES HERE")
    # From here, we initiate a starting state, and just simulate trajectories with the trained dynamics as well as the expert dynamics.

    stateerrorarr1 = np.zeros((100, TRAJ_LENGTH))
    stateerrorarr2 = np.zeros((100, TRAJ_LENGTH))
    actionerrorarr = []
    paramrelerror1 = np.zeros((100, TRAJ_LENGTH))
    paramrelerror2 = np.zeros((100, TRAJ_LENGTH))
    meancxoverall = []
    meancxexpert = []
    experttraj = np.zeros((TRAJ_LENGTH+10, 3))
    simulatedtrajoverall = np.zeros((TRAJ_LENGTH+10, 3))

 
    for k in range(100):
        obs = Tensor(system.generate_initial_obs())
        print("Initial Obs = " + str(obs))

        
        controlinputs = []

        # Generating Expert Trajectory
        actualinitialobs = obs.cpu()
        for trajstep in range(TRAJ_LENGTH):
            simulatedcxarr = np.zeros((TEST_NUM_STEPS, 2))
            expertcxarr = np.zeros((TEST_NUM_STEPS, 2))
            exp_initialobs = actualinitialobs
            sim_initialobs = actualinitialobs
            simulatedtraj = np.zeros((TEST_NUM_STEPS, 3))

            # print("Done with the Expert Trajectories")
            # Generating Simulated Trajectory
            # sim_initialobs = sim_initialobs.to(device='cuda')
            sim_initialobs = torch.Tensor(sim_initialobs).cuda()
            # sim_initialobs = torch.from_numpy(sim_initialobs).to(device='cuda')
            for step in range(TEST_NUM_STEPS):
                simulatedtraj[step] = np.array(sim_initialobs.cpu())
                simulatedtrajoverall[trajstep+step] = np.array(sim_initialobs.cpu())
                initialstate = (sim_initialobs[0:2])
                value, action, _, _, _ = actor_critic.act(
                            sim_initialobs, GAIL_DIS_HDIM,
                            np.array([False, False]))
                action = action[np.newaxis, :]
                # next_state, next_obs, _ = system.step_with_predefined_actions(initialstate, controlinputs[step+1], action)
                next_state, next_obs, simulatedcx = system.step(initialstate, action)
                sim_initialobs = Tensor(next_obs)
                simulatedcxarr[step] = simulatedcx
            
            for step in range(TEST_NUM_STEPS):
                experttraj[trajstep+step] = np.array(exp_initialobs)
                initialstate = (exp_initialobs[0:2])
                initialstate = torch.Tensor(initialstate).cuda()
                next_state, next_obs, expertcx = system.step(initialstate, None)
                expertcxarr[step] = expertcx
                controlinputs.append(np.array(next_obs[2]))
                exp_initialobs = next_obs
            actualinitialobs = experttraj[trajstep+1]
            
            #Calculating difference in the trajectory states and trajectory actions
            trajerror = (simulatedtraj[:,0:2] - experttraj[trajstep:trajstep+TEST_NUM_STEPS, 0:2])/experttraj[trajstep:trajstep+TEST_NUM_STEPS, 0:2]
            paramerror = (simulatedcxarr-expertcxarr)/expertcxarr
            meancxoverall.append(np.mean(simulatedcxarr))
            meancxexpert.append(np.mean(expertcxarr))
            # pdb.set_trace()
            statenorm1 = np.linalg.norm((trajerror[:,0]), 2)/TEST_NUM_STEPS
            statenorm2 = np.linalg.norm((trajerror[:,1]), 2)/TEST_NUM_STEPS
            paramerrornorm1 = np.linalg.norm(paramerror[:,0], 2)/TEST_NUM_STEPS
            paramerrornorm2 = np.linalg.norm(paramerror[:,1], 2)/TEST_NUM_STEPS
            # pdb.set_trace()
            # statenorm = np.linalg.norm((trajerror[:,0:2]), 2)/TEST_NUM_STEPS
            # actionnorm = np.linalg.norm(trajerror[:,2], 2)/TEST_NUM_STEPS

            stateerrorarr1[k,trajstep] = statenorm1
            stateerrorarr2[k,trajstep] = statenorm2
            paramrelerror1[k,trajstep] = paramerrornorm1
            paramrelerror2[k,trajstep] = paramerrornorm2
        # actionerrorarr.append(actionnorm)
        axs[0][1].cla()
        axs[0][1].set_title("SIM. Vs. EXP. TRAJ.")
        axs[0][1].plot(simulatedtrajoverall[:,0], simulatedtrajoverall[:,1], color='red', label='Simulated Traj')
        axs[0][1].plot(experttraj[:,0],experttraj[:,1],color='blue', label="Expert Traj" )
        plt.pause(0.005)

    stateerrorarr1 = np.linalg.norm(stateerrorarr1, 2, axis=0)/100
    stateerrorarr2 = np.linalg.norm(stateerrorarr2, 2, axis=0)/100
    paramrelerror1 = np.linalg.norm(paramrelerror1, 2, axis=0)/100
    paramrelerror2 = np.linalg.norm(paramrelerror2, 2, axis=0)/100

    # stateerrorarr1 = savgol_filter(stateerrorarr1, 5, 2)
    # stateerrorarr2 = savgol_filter(stateerrorarr2, 5, 2)
    # paramrelerror1 = savgol_filter(paramrelerror1, 5, 2)
    # paramrelerror2 = savgol_filter(paramrelerror2, 5, 2)

    axs[0][2].set_title("x1 and x2 ERR. (Sim. vs. Exp.")
    axs[0][2].plot(stateerrorarr1)
    axs[0][2].plot(stateerrorarr2)
    axs[1][2].plot(paramrelerror1)
    axs[1][2].plot(paramrelerror2)
    # axs[0][3].set_title("Mean Err. - Distr. (Sim. vs. Exp.")
    axs[1][3].set_title("Var Err. - Distr. (Sim. vs Exp.")
    merrorarr = np.array(merrorarr)
    verrorarr = np.array(verrorarr)
    # print(merrorarr)
    # print(verrorarr)
    # axs[0][3].boxplot(merrorarr)
    axs[1][3].boxplot(verrorarr)
    print("------------------------------")
    print("MEAN ERROR IN X IS = " + str(np.mean(stateerrorarr1)) + ", " + str(np.mean(stateerrorarr2)))
    print("MEAN ERROR IN C IS = " + str(np.mean(paramrelerror1)) + ", " + str(np.mean(paramrelerror2)))
    print("MEAN C IS = " + str(np.mean(meancxoverall)))
    print("AND THE SAME FOR EXPERT = " + str(np.mean(meancxexpert)))
    print("------------------------------")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
