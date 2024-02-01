# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

#from ml_logger import logger

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # logger.log_params(Args = vars(train_cfg))
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.headless = True
    train(args)


# Summary training
# logs/flat_go1/Jan12_11-17-01_ : training with every parameters default
# logs/flat_go1/Jan12_15-38-30_ : train with hip_reduction_scale added in legged_robot.py - no big changes 
# Keep hip reduction scale from this point
# logs/flat_go1_act_net/Jan12_16-23-54_: train with actuator network from walk-these-ways - works well
# Only use actuator net now
# logs/flat_go1_act_net/Jan15_10-47-31_: train with 48 observations as it represents the reality - simulations is good
# logs/flat_go1_act_net/Jan16_18-51-39_: removed base lin vel and projected gravity - robot in a crouched position but it's still walking
# logs/flat_go1/Jan17_09-57-43_ : train without actuator network (with P)and without hip_scale_reduction with num_obs = 48 to see if it has an impact on the real robot - very shitty
# logs/flat_go1/Jan17_10-51-07_: use go1 urdf from Stoch and same config file : SHITTY

# Back to start
# logs/flat_go1/Jan17_11-02-13_ : train with everything default num_obs = 235 - WORKS PERFECTLY
# logs/flat_go1/Jan17_11-28-32_ : train with num_obs = 48 ans measure_heights = False (accordingly) - it does WORK but the robot is in a crouched position
# logs/flat_go1/Jan17_11-56-54_ : (NOISE DEFAULT) + DOMAIN RAND modify friction_range [0.05, 4.5], randomize_base_mass = True, added_mass_range = [-1., 3.], - NO CROUCHED POSITION ANYMORE
# logs/flat_go1/Jan17_12-22-15_ : don't scale the observations - the robot is a bit more crouched but that's okay
# logs/flat_go1/Jan17_13-24-09_ : modified rewards torques = -0.0001 # -0.0002 - works great
                                                # dof_pos_limits = -10.0
                                                # action_rate = -0.01
                                                # orientation = -5.
                                                # base_height = -30.
# logs/flat_go1/Jan17_13-50-22_ : delete base_lin_vel in the observation space - DEPLOYED ON THE ROBOT - joint vel causing some troubles
# logs/flat_go1/Jan17_16-22-28_ : train in torque space  control_type = 'T' - CRASHED
# logs/flat_go1/Jan17_16-41-12_ : train in torque space  control_type = 'T' - PURE SHIT

# THE BEHAVIOR BETWEEN logs/flat_go1/Jan17_13-50-22_and reality is the SAME - need to retune the reward function !
# logs/flat_go1/Jan17_17-38-19_ : let's retake the original reward function scales : torques = -0.0002 , dof_pos_limits = -10.0 - the robot still has some difficulties to perfectly stands and is penché vers l'avant
# logs/flat_go1/Jan18_12-37-59_: add more noise to dof vel (dof_vel = 4), keep reward function from logs/flat_go1/Jan17_13-24-09_, increased number of iterations to 2000 - too much 
# logs/flat_go1/Jan18_13-57-52_ : noise dof vel = 2.5 and add scales to the observations (default one) - lin_vel = 1.0 # 2.0
        # ang_vel = 0.25 # 0.25
        # dof_pos = 1.0 # 1.0
        # dof_vel = 0.05 # 0.05 ------------------------> not converging anymore
# logs/flat_go1/Jan18_14-32-55_ : remove obs scales and just keep dof_vel noise = 2.5 and add more mass range [-1., 3.] - DEPLOYED ON THE ROBOT - Works a bit better !!
# logs/flat_go1/Jan31_10-58-33_ : dof_vel_noise = 2.5, reput obs scales for ang_vel and dof_vel and see if training converges - DOES NOT CONVERGE
# logs/flat_go1/Jan31_11-24-39_ : try with dof vel noise scale = 1.5 and keep obs scale to see if it converges - DOES NOT CONVERGE
# : just keep dof:vel obs scale = 0.05, all other to 1 : see if converges or not : otherwise reput obs scales everywhere to 1 - DID not converge so scales = 1 everywhere

# Next steps are conducted to observe if the actuator network plays a role in sim-to-real transfer
# : just add actuator network to see how it behaves - DEPLOYED ON THE ROBOT WAY BETTER THAN PREVIOUSLY - now focus on making the training pipeline better