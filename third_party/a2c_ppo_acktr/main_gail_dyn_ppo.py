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

# from my_pybullet_envs import utils as gan_utils

import logging
import sys

TRAJECTORY_LOAD_PATH1="./data/data/"
TRAJECTORY_LOAD_PATH2= "./data/parameters"
TRAJECTORY_LENGTH = 100
TRAJECTORIES_NUM = 20 # MAX 90 
BATCH_SIZE = 80
HIDDEN_SIZE = 128
EPOCHS = 1000

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
    # sasdata = np.array(sasdata)
    # print(sasdata[0])
    # print(sasdata[1])
    # print(sasdata[2])
    print("PRINTING SHAPES OF THE TWO DATA INPUTS - ")
    print(np.shape(xcommau))
    print(np.shape(paramsdata))
    # # print(sasdata.iloc[0])
    # print(np.array(sasdata.iloc[1]))
    sasdata = np.zeros((np.shape(xcommau)[0], np.shape(xcommau)[1] + 2))
    for i in range(np.shape(xcommau)[0]-1):
        sasdata[i, :] = np.concatenate((xcommau[i], np.array([xcommau[i+1,0]]), np.array([xcommau[i+1,1]])))
    
    sasdata = sasdata[:-1]
    print(np.shape(sasdata))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    # save_path = os.path.join(args.save_dir, args.algo)

    torch.set_num_threads(4)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    actor_critic = SplitPolicy(3, 2)
    actor_critic.to(device)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # file_handler = logging.FileHandler("{0}/{1}.log".format(save_path, "console_output"))
    # file_handler.setFormatter(log_formatter)
    # root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    
    expertdataset = TensorDataset(Tensor(np.array(sasdata)))
    drop_last = len(expertdataset) > BATCH_SIZE
    gail_train_loader = DataLoader(expertdataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last)
    
    # # expert_sas_w_past = gan_utils.load_sas_wpast_from_pickle(
    # #     args.gail_traj_path,
    # #     downsample_freq=int(args.gail_downsample_frequency),
    # #     load_num_trajs=args.gail_traj_num
    # # )
    # # # assume in the order of s_old,..., a_old,..., st+1
    s_dim = 2
    a_dim = 1
    s_idx = np.array([0])
    a_idx = np.array([0])

    # # info_length = len(s_idx) * s_dim + len(a_idx) * a_dim + s_dim       # last term s_t+1   # 2 + 1 + 2 (S + a + S)
    # Defining the Discriminator 
    discr = gail.Discriminator(
        5, args.gail_dis_hdim,
        device)
    # # expert_merged_sas = gan_utils.select_and_merge_sas(expert_sas_w_past, a_idx=a_idx, s_idx=s_idx)
    # # assert expert_merged_sas.shape[1] == info_length
    # # expert_dataset = TensorDataset(Tensor(expert_merged_sas))

    gail_tar_length = np.shape(sasdata)[0] * 1.0 / args.gail_traj_num * args.gail_downsample_frequency
    # print("GAIL TAR LENGTH = " + str( np.shape(sasdata)[0]))

    # # drop_last = len(expert_dataset) > args.gail_batch_size
    # # gail_train_loader = DataLoader(
    # #     expert_dataset,
    # #     batch_size=args.gail_batch_size,
    # #     shuffle=True,
    # #     drop_last=drop_last)
    # #####################################################################
    # obs = envs.reset()
    system = System()

    obs = Tensor(system.generate_initial_obs())
    print("First Obs = " + str(obs))

    # Creating empty object to store state,action values#################
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
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
    max_num_episodes = args.num_episodes if args.num_episodes else np.infty

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    from third_party.a2c_ppo_acktr.baselines.common.running_mean_std import RunningMeanStd
    ret_rms = RunningMeanStd(shape=())
    # errorcx = []
    mainerrorarr = []
    valuelossarr = []
    actionlossarr = []
    avgparamerror = []
    fig, axs = plt.subplots(2 , 1)
    # fig2, axs2 = plt.subplots(2 , 1)
    
    
    # Start main loop ###############
    while j < num_updates and total_num_episodes < max_num_episodes:
        
        # Learning rate decay for the main loop
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        errorcxexpert = []
        paramsarr = []
        # plt.clf()
        ##BEGIN###############################################################
        # INITALLY COLLECTING TRAJECTORIES FROM CURRENT POLICY FROM THE SIMULATOR 
        for step in range(args.num_steps):
            # print(args.num_steps) 300*8
            # Sample actions
            # print("RAN ONCE")
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            
            # print("SHAPE OF rollout obs = " + str(np.shape(rollouts.obs[step])))
            # Obser reward and next obs
            # obs, reward, done, infos = envs.step(action)
            done1 = 1
            done2 = 1

            next_state, next_obs, errorcx = system.step(rollouts.obs[step][0][0:2], action)
            # errorcxexpert.append(errorcx)
            # followernextstate = follower.step(obs[2:4], obs[0:2], action)
            #TODO: REWARD SHAPING
            reward = Tensor(1) #-((leadernextstate[0] - followernextstate[0] - 20) + (leadernextstate[1] - followernextstate[1]))
            next_state = Tensor(next_state.cuda())
            next_obs = Tensor(next_obs).cuda()

            paramsarr.append(action.cpu().numpy())

            # print(np.shape(obs))
            # print(np.shape(next_state[:,0]))
            sas_feat = torch.cat((rollouts.obs[step][0], next_state[:,0])) #Tensor(np.concatenate(np.array(obs),np.array(next_state)))
            # print("SAS_FEAT SHAPE = " + str(sas_feat.shape))
            if (next_state[0] < 30 and next_state[0] > -30) and np.abs(next_obs[2].cpu().numpy()) < 50:
                done1 = 0
            if (next_state[1] < 30 and next_state[1] > -30) and np.abs(next_obs[2].cpu().numpy()) < 50:
                done2 = 0

            # sas_feat = np.zeros((args.num_processes, 5))
            # for core_idx, info in enumerate(infos):
            #     if 'episode' in info.keys():
            #         episode_rewards.append(reward)
            #     # get the past info here and apply filter
            #     # sas_info = info["sas_window"]
            #     sas_feat[core_idx, :] = gan_utils.select_and_merge_sas(sas_info, s_idx=s_idx, a_idx=a_idx)

            # print(sas_feat)
            # print("###################################")
            # # If done then clean the history of observations.
            # #TODO: Figuring out done with respect to the reward. 
            masks = Tensor(
                [done1, done2])
            bad_masks = Tensor(
                [1.0, 1.0])
            rollouts.insert(next_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, Tensor(sas_feat))

        ##END####################################################################
        mainerrorarr.append(np.mean(errorcxexpert))
        # print("Finished Collecting Simulator trajectories")
        avgparamerror.append(np.mean(np.abs(paramsarr)))
        ##BEGIN#################################################################
        # Getting the next value of the SIMULATOR FUNCTION F
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        gail_loss, gail_loss_e, gail_loss_p = None, None, None
        gail_epoch = args.gail_epoch


        # use next obs feat batch during update...
        # if j % 2 == 0:
        #TODO: EVERYTHING ALRIGHT UPTIL HERE
        # uses the expert trajs and collected trajs to update weights and give loss ###############
        for _ in range(gail_epoch):
            gail_loss, gail_loss_e, gail_loss_p = discr.update_gail_dyn(gail_train_loader, rollouts)
    #     ##################################################################################
        # print("Finished Updating the Discriminator")
    # #     ##BEGIN###############################################################################
    #     # This block of code is used to implement the staying-alive bonus. 
        
        num_of_dones = (1.0 - rollouts.masks).sum().cpu().numpy() \
            + args.num_processes / 2
        # print(num_of_dones)
        num_of_expert_dones = (args.num_steps * args.num_processes) / gail_tar_length
        # print(num_of_expert_dones)

        # d_sa < 0.5 if pi too short (too many pi dones),
        # d_sa > 0.5 if pi too long
        d_sa = 1 - num_of_dones / (num_of_dones + num_of_expert_dones)
        # print(d_sa)
        if args.no_alive_bonus:
            r_sa = 0
        else:
            r_sa = np.log(d_sa) - np.log(1 - d_sa)  # d->1, r->inf
    #     ##END###################################################################
        # print("Finished Updating the Staying Alive Bonus")
    
    #     ##BEGIN#################################################################file_handler = logging.FileHandler("{0}/{1}.log".format(save_path, "console_output"))
    # file_handler.setFormatter(log_formatter)
    # root_logger.addHandler(file_handler)
    #     # use next obs feat to overwrite reward...
    #     # overwriting rewards by gail

    #     # This block computes the rewards that are passed onto the generator function (By updating the rollouts variable).
        # The rewards are computed using the discriminator class.  
        
        for step in range(args.num_steps):
            rollouts.rewards[step], returns = \
                discr.predict_reward_combined(
                    rollouts.obs_feat[step + 1], args.gamma,
                    rollouts.masks[step], offset=-r_sa
                )
            # print("Can do 1st step")
            # print(rollouts.obs[step, 0])
            # print(rollouts.obs_feat[step+1, 0])

            # redo reward normalize after overwriting
            # print(rollouts.rewards[step], returns)
            ret_rms.update(returns.view(-1).cpu().numpy())
            rews = rollouts.rewards[step].view(-1).cpu().numpy()
            rews = np.clip(rews / np.sqrt(ret_rms.var + 1e-7),
                           -10.0, 10.0)
            # print("Can do 2nd step")
            # print(ret_rms.var)    # just one number
            rollouts.rewards[step] = Tensor(rews).view(-1, 2)
            # print("after", rollouts.rewards[step], returns)
            # print("Can do 3rd step")
            # final returns
            # print(returns)
            gail_rewards.append(torch.mean(returns).cpu().data)
            # print("Can do 4th step.")
    #     ##END####################################################################
        # print("Updating Rollouts done")
    #     ## COMPUTE_RETURNS FUNCTION ##########################################
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, not args.no_proper_time_limits)
    #     ######################################################################
        # print("Computing returns in rollouts done")
    #     # This line updates the simulator parameter function f from thetai to thetai+1 using PPO and the rewards computed
    #     # (that were updated in rollouts just above)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        # print("Updating Agent Done")
    #     ### AFTER_UPDATE FUNCTION ############################################
    #     # This function just updates the first value in multiple arrays to be the last value. (Because the next episode starts from there)
        rollouts.after_update()
    #     ######################################################################


        # # save for every interval-th episode or for the last epoch
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
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
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

        print((position1).shape)
        l_to_show = position1.shape

        # axs[0].set_xlim([-50, 50])
        # axs[0].set_ylim([-50, 50])
        # axs[1].set_xlim([-5, 5])
        # axs[1].set_ylim([-5, 5])
        axs[0].cla()
        axs[1].cla()
        axs[0].plot(position1.cpu().numpy(), color='red', label='Predicted')
        axs[0].plot(xcommau[:250,0], color='blue', label='Expert')
        axs[1].plot(position2.cpu().numpy(), color='red')
        axs[1].plot( xcommau[:250,1], color='blue')
        # axs[0].plot(valuelossarr)
        axs[0].set_title("Plot Position 1")
        axs[1].set_title("Plot Position 2")
        axs[0].legend()
        # axs[1].plot(position2.cpu().numpy())
        # print(np.shape(paramsarr))
        # axs[1].plot(np.array(paramsarr)[:,0,0], np.array(paramsarr)[:,0,1], color='red', label='Predicted')
        # axs[1].plot(paramsdata[:,0], paramsdata[:,1], color='blue', label='Expert')
        # axs[1].set_title("Params")
        # axs[1].legend()
        # axs[2].plot(delta_predition)
        # axs[2].set_title("Delta Prediction")
        plt.pause(0.05)
        
    fig, axs2 = plt.subplots(2 , 1)
    axs2[0].plot(avgparamerror)
        
    plt.show()


if __name__ == "__main__":
    main()
