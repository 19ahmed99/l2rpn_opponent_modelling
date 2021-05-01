# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg
from DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from prioritized_replay_buffer import PrioritizedReplayBuffer



class DoubleDuelingDQN(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 is_training=False):
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)
        self.obs_space = observation_space

        # Filter
        #print("Actions filtering...")
        self.action_space.filter_action(self._filter_action)
        #print("..Done")

        self.action_path = "./allactions.npy"
        self.converter = IdToAct(self.action_space)
        self.converter.init_converter()
        self.converter.save(*os.path.split(self.action_path))
        self.all_actions = np.array(self.converter.all_actions).tolist()
        self.all_acts_dict = {tuple(el.to_vect().tolist()): i for i, el in enumerate(self.all_actions)}

        # Store constructor params
        self.name = name
        self.num_frames = cfg.N_FRAMES
        self.is_training = is_training
        self.batch_size = cfg.BATCH_SIZE
        self.lr = cfg.LR
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []

        # Declare training vars
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_rewards_moving_avg = None
        self.losses = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = 0.0

        # Store the current opponent action
        self.opponent_action = None
        # List of all opponent actions
        self.all_opponent_actions = []
        # Stores opponent next action
        self.opponent_next_action = None
        # List of all opponent next actions
        self.all_opponent_next_actions = []
        self.count_non_do_nothing_opp_act = 0 

        # Compute dimensions from intial spaces
        self.observation_size = self.obs_space.size_obs()
        self.action_size = self.action_space.size()
        self.action_vect_size = 258
        
        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=cfg.LR_DECAY_STEPS,
                                         learning_rate_decay_rate=cfg.LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False
            
    def _init_training(self):
        self.epsilon = cfg.INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_rewards_moving_avg = []
        self.losses = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(cfg.PER_CAPACITY, cfg.PER_ALPHA)
        self.Qtarget = DoubleDuelingDQN_NN(self.action_size,
                                           self.observation_size,
                                           num_frames = self.num_frames)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = []
        if self.is_training:
            self.frames2 = []

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    def _adaptive_epsilon_decay(self, step):
        ada_div = cfg.DECAY_EPSILON / 10.0
        step_off = step + ada_div
        ada_eps = cfg.INITIAL_EPSILON * -math.log10((step_off + 1) / (cfg.DECAY_EPSILON + ada_div))
        ada_eps_up_clip = min(cfg.INITIAL_EPSILON, ada_eps)
        ada_eps_low_clip = max(cfg.FINAL_EPSILON, ada_eps_up_clip)
        return ada_eps_low_clip
            
    def _save_hyperparameters(self, logpath, env, steps):

        r_instance = env._reward_helper.template_reward
        hp = {
            "lr": cfg.LR,
            "lr_decay_steps": cfg.LR_DECAY_STEPS,
            "lr_decay_rate": cfg.LR_DECAY_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "stack_frames": cfg.N_FRAMES,
            "iter": steps,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "per_alpha": cfg.PER_ALPHA,
            "per_beta": cfg.PER_BETA,
            "per_capacity": cfg.PER_CAPACITY,
            "update_freq": cfg.UPDATE_FREQ,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):
        li_vect=  []
        for el in observation.attr_list_vect:
            v = observation._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)
            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            li_vect.append(v_res)
        return np.concatenate(li_vect)

    def convert_act(self, action):
        return super().convert_act(action)

    ## Baseline Interface
    def reset(self, observation):
        self._reset_state(observation)
        self._reset_frame_buffer()

    def my_act(self, state, reward, done=False):
        # Register current state to stacking buffer
        self._save_current_frame(state)
        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return 0 # Do nothing
        # Infer with the last num_frames states
        a, _ = self.Qmain.predict_move(np.array(self.frames), np.array(self.all_opponent_actions[-self.num_frames:]))        
        
        return a

    def act(self, obs, reward, done):
        self.obs = obs     
        # Store opponent action
        self.store_opponent_action(obs)
        transformed_observation = self.convert_obs(obs)
        encoded_act = self.my_act(transformed_observation, reward, done)
        return self.convert_act(encoded_act)

    
    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    # Store current opponents action
    # This function is used as an alterantive way to retrieve the opponent action with using the info variable returned by the step function
    # This function can be used when the agent is training or not
    def store_opponent_action(self, obs):
        # A Do_Nothing action
        opponent_action = self.action_space({})

        # Get all the powerline id that will require maintenance
        maintenance = obs.time_next_maintenance
        maintenance_powerline_id = [i for i in range(len(maintenance)) if maintenance[i] != -1]
        
        # Retrive all the cooldown_duration for disconnected powerlines that are equal to 47 timesteps
        cooldown_duration = obs.time_before_cooldown_line
        cooldown_powerline_id = [i for i in range(len(cooldown_duration)) if cooldown_duration[i] == 47]
        
        for pid in cooldown_powerline_id:
            # Check if it is disconnected due to a maintenacance or an attack
            if pid in maintenance_powerline_id and maintenance[pid] == 0:
                cooldown_powerline_id.remove(pid)      
            else:
                powerline_attacked = pid
                opponent_action = self.action_space({"change_line_status": [int(powerline_attacked)]})
                self.count_non_do_nothing_opp_act += 1
        
        # Convert the opponent action to its vector representation
        opp_act_as_vect = (self.converter() + opponent_action).to_vect()
        self.opponent_action = opp_act_as_vect
        self.all_opponent_actions.append(self.opponent_action)

    # Store the opponent action using the info variable returned by the step() function which give information about the next observation
    # This function can only be used during the training
    def store_opponent_next_action(self, info):
        opponent_action = self.action_space()
        attack_duration = info["opponent_attack_duration"]

        if attack_duration == 48:
            powerline_attacked = np.where(info["opponent_attack_line"])[0]
            # Let the opponent action be a powerline disconnection action of the powerline attacked
            opponent_action = self.action_space({"change_line_status": [int(powerline_attacked)]})
        
        # Convert the opponent action to its vector representation
        opp_act_as_vect = (self.converter() + opponent_action).to_vect()
        self.opponent_next_action = opp_act_as_vect
        self.all_opponent_next_actions.append(self.opponent_action)


    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              num_pre_training_steps=0,
              logdir = "logs-train"):
        # Make sure we can fill the experience buffer
        if num_pre_training_steps < self.batch_size * self.num_frames:
            num_pre_training_steps = self.batch_size * self.num_frames

        # Loop vars
        num_training_steps = iterations
        num_steps = num_pre_training_steps + num_training_steps
        self.epsilon = cfg.INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        self.done = True
        step = 0 

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".h5")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)

        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                new_obs = env.reset() # This shouldn't raise
                self.reset(new_obs)
            if cfg.VERBOSE and step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, self.epsilon))

            # Save current observation to stacking buffer
            self._save_current_frame(self.state)

            # Store opponent current action
            self.store_opponent_action(new_obs)

            # Choose an action
            if step <= num_pre_training_steps:
                a = self.Qmain.random_move()
            elif np.random.rand(1) < self.epsilon:
                a = self.Qmain.random_move()
            elif len(self.frames) < self.num_frames:
                a = 0 # Do nothing
            else:
                a, _ = self.Qmain.predict_move(np.array(self.frames), np.array(self.all_opponent_actions[-self.num_frames:]))

            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            new_state = self.convert_obs(new_obs)
            # if info["is_illegal"] or info["is_ambiguous"] or \
            #    info["is_dispatching_illegal"] or info["is_illegal_reco"]:
            #     # if cfg.VERBOSE:
                    # print (a, info)

            # Store opponent next action
            self.store_opponent_next_action(info)

            # Save new observation to stacking buffer
            self._save_next_frame(new_state)

            # Save to experience buffer
            if len(self.frames2) == self.num_frames:
                self.per_buffer.add(np.array(self.frames),
                                    a, np.array(self.all_opponent_actions[-4:]),
                                    reward,
                                    np.array(self.frames2), np.array(self.all_opponent_next_actions[-4:]),
                                    self.done)

            # Perform training when we have enough experience in buffer
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Decay chance of random action
                self.epsilon = self._adaptive_epsilon_decay(training_step)

                # Perform training at given frequency
                if step % cfg.UPDATE_FREQ == 0 and \
                   len(self.per_buffer) >= self.batch_size:
                    # Perform training
                    self._batch_train(training_step, step)

                    if cfg.UPDATE_TARGET_SOFT_TAU > 0.0:
                        tau = cfg.UPDATE_TARGET_SOFT_TAU
                        # Update target network towards primary network
                        self.Qmain.update_target_soft(self.Qtarget.model, tau)

                # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
                   step % (cfg.UPDATE_FREQ * cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    self.Qmain.update_target_hard(self.Qtarget.model)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                current_reward_moving_avg = sum(self.epoch_rewards)/len(self.epoch_rewards)
                self.epoch_rewards_moving_avg.append(current_reward_moving_avg)
                self.epoch_alive.append(alive_steps)
                if cfg.VERBOSE:
                    print("Survived [{}] steps".format(alive_steps))
                    print("Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1
            
            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                modelpath = os.path.join(save_path, self.name + str(step) +".h5")
                self.save(modelpath)


            # Iterate to next loop
            step += 1
            # Make new obs the current obs
            self.obs = new_obs
            self.state = new_state


        # Save model after all steps
        modelpath = os.path.join(save_path, self.name + str(step) +".h5")
        self.save(modelpath)

        print("Number of opponent action that are not do_nothing :  {} ".format(self.count_non_do_nothing_opp_act))

        return self.epoch_rewards, self.epoch_rewards_moving_avg, self.losses



    def _batch_train(self, training_step, step):
        """Trains network to fit given parameters"""

        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, cfg.PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        opp_a_batch = sample_batch[2]
        r_batch = sample_batch[3]
        s2_batch = sample_batch[4]
        opp_next_a_batch = sample_batch[5]
        d_batch = sample_batch[6]
        w_batch = sample_batch[7]
        idx_batch = sample_batch[8]

        Q = np.zeros((self.batch_size, self.action_size))
        
        
        input_s_size = self.observation_size * self.num_frames 
        input_opp_size = self.action_vect_size * self.num_frames

        # Reshape frames to 1D
        input_s_t = np.reshape(s_batch, (self.batch_size, input_s_size))
        input_opp_t = np.reshape(opp_a_batch, (self.batch_size, input_opp_size))
        input_s_t_1 = np.reshape(s2_batch, (self.batch_size, input_s_size))
        input_opp_t_1 = np.reshape(opp_next_a_batch, (self.batch_size, input_opp_size))

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.Qmain.model.predict([input_s_t, input_opp_t], batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.Qmain.model.predict([input_s_t_1,input_opp_t_1], batch_size=self.batch_size)
        Q2 = self.Qtarget.model.predict([input_s_t_1,input_opp_t_1], batch_size=self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += cfg.DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.train_on_batch([input_s_t, input_opp_t], Q, w_batch)
        self.losses.append(loss)

        # Update PER buffer
        priorities = self.Qmain.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)

        # Log some useful metrics every even updates
        if step % (cfg.UPDATE_FREQ * 2) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)
                tf.summary.scalar("lr", self.Qmain.train_lr, step)
            if cfg.VERBOSE:
                print("loss =", loss)
