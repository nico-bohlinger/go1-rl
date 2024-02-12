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
# logs/flat_go1_act_net/Feb01_09-20-17_ : just add actuator network to see how it behaves - DEPLOYED ON THE ROBOT WAY BETTER THAN PREVIOUSLY - now focus on making the training pipeline better
# logs/flat_go1_act_net/Feb01_14-19-36_ : add com displacement in domain randomization - DEPLOYED ON THE ROBOT - WORK BETTER
# logs/flat_go1_act_net/Feb01_15-13-39_ : dof vel obs scales = 0.5 (maximum value before the robot vibrates in reality) - the robot still vibrates so for now let's forget about observation scales
# logs/flat_go1_act_net/Feb01_15-57-07_ : increase feet air time in the reward function to 1.5 - DID NOT CONVERGE
# logs/flat_go1_act_net/Feb02_09-29-39_ : retry with feet_air_time = 1.2 - did not see drastic improvement
# logs/flat_go1_act_net/Feb02_10-37-30_ : train with heading= False and lin_vel ranges = [-0.6, 0.6] - DID NOT CONVERGE - remove commands change 
# logs/flat_go1_act_net/Feb02_11-25-29_ : introduce reward feet slip with -0.04 - DEPLOYED ON THE ROBOT - wayyy better
# logs/flat_go1_act_net/Feb02_11-58-56_ : feet_slip = -0.08 - i think it's better with -0.04
# logs/flat_go1_act_net/Feb02_14-45-23_ : add action smoothness 1 and 2 penalties (joint_pos_target was not properly initialized - hope it's okay) - mean reward = 0 all time
# logs/flat_go1_act_net/Feb02_14-53-03_ : remove action_smoothness 1 & 2 to see if it was due to joint_pos_target not properly initialized - OK it's good (i cancel)
# logs/flat_go1_act_net/Feb02_14-59-07_ : add only action_smoothness 1 - DEPLOYED ON THE ROBOT - looks maybe better but difficult to judge 

#Time to add CERN specific environmental impact on robot (magnetic field, radiation)
# logs/flat_go1_act_net/Feb05_11-06-39_ : start with lag_timesteps - see it it converges 
# logs/flat_go1_act_net/Feb05_11-47-36_ : readd action smoothness 2 because maybe it needed more time -  NOT WORKING
# logs/flat_go1_act_net/Feb05_14-38-19_ : forgot to reset last_last_actions - is it why it never converged ? + continue with Kp and Kd offsets (motor offset) - does not look good (CANCELLED)
# logs/flat_go1_act_net/Feb05_14-41-52_ : remove action smoothness 2 - only keep randomize dof props (motor offset, strengths, etc) - forgot something in init_custom_buffers
# logs/flat_go1_act_net/Feb05_15-43-19_ : retry with motor offsets and strengths (Kp and Kd are only for the P control type) - shiiiiit (STOPPED)
# logs/flat_go1_act_net/Feb05_16-06-27_ : back to when there was lag timesteps only and add com_pos_range = [-0.15, 0.15] - not convergiiing (STOPPED)
# logs/flat_go1_act_net/Feb05_16-16-20_ : back to com_pos_range = [-0.1, 0.1] - works again - PC CRASHED

# logs/flat_go1_act_net/Feb05_16-46-43_ : add randomize_gravity - DOES NOT CONVERGE maybe I have to wait 10k iterations instead of 3k : https://github.com/Improbable-AI/walk-these-ways/issues/8 or https://github.com/Improbable-AI/walk-these-ways/issues/22
# logs/flat_go1_act_net/Feb05_17-08-02_ : tried with gravity_range = [-0.0, 0.0] - does not look great right in the beginning - good again
# logs/flat_go1_act_net/Feb05_17-11-00_ : remove gravity_range - good again
# logs/flat_go1_act_net/Feb05_17-15-32_ : try with small gravity_range gravity_range = [-0.1, 0.1] - definitely does not work
# logs/flat_go1_act_net/Feb05_17-19-40_ : try without pushes to the robot - does not converge 

# logs/flat_go1_act_net/Feb05_17-41-12_ : try with trimesh - not converging
# logs/flat_go1_act_net/Feb05_17-45-59_ : try with feet_air_time =  0.01 and dof_acc = -2.5e-7 - 40k iterations - not good at the end
# logs/flat_go1_act_net/Feb06_08-50-06_ : retry without feet_air_time - see if it converges after 40k iterations
# logs/flat_go1_act_net/Feb06_17-47-07_ : retry gravity randomization by removing randomize_gravity in create_envs() and push_robots




# Try adding observation history like Jai did 
# logs/flat_go1_act_net/Feb07_14-51-09_ : try addind the observation history with parameters from logs/flat_go1_act_net/Feb06_17-47-07_ - FORGOT TO RESIZE NOISE VEC !!!!
# logs/flat_go1_act_net/Feb07_16-40-21_ : try with push_robots = True and friction range = [0.05, 1.25]
# logs/flat_go1_act_net/Feb07_17-21-42_ : retry with adding randomize_gravity in create_envs() - still works
# logs/flat_go1_act_net/Feb08_09-09-37_ : remove action smoothness 1 and put action smoothness 2 - definetely does not work
# logs/flat_go1_act_net/Feb08_09-33-56_ : try with base heigth = -20 - shiit back to -30


# Following dream waq implementation with privileged information and same reward function
# logs/flat_go1_act_net/Feb08_10-08-40_ : see if it converges first and test (it's on plane terrain now) - does not work 

# logs/flat_go1_act_net/Feb08_11-08-26_ : add feet clearnace penalty -0.01 - we see that the rear foot lifts upper

# logs/flat_go1_act_net/Feb08_12-03-23_ : Retry adding privileged information with obs buf + height + contact forces + base lin vel - looks way better
# logs/flat_go1_act_net/Feb08_12-35-57_ : retry with orientation = -0.2 - not good 
# logs/flat_go1_act_net/Feb08_14-25-22_ : retry with torques = -0.00001 - better before
# logs/flat_go1_act_net/Feb08_14-53-19_ : add joint power and power distribution penalties -BETTER - DEPLOYED ON THE ROBOT
# logs/flat_go1_act_net/Feb08_16-01-51_ : try with trimesh - does not seem to rise 
# logs/flat_go1_act_net/Feb08_16-39-55_ : remove stairs up and down and increase discrete proportions

# logs/flat_go1_act_net/Feb08_17-28-45_ : gravity_range = [-1, 1] - looks wayyyy better
# logs/flat_go1_act_net/Feb09_09-40-02_ : try with motor offset and strength randomization - does not work

# Rough Go1 
# logs/rough_go1/Feb09_10-10-04_ : try with full terrain - does not seem to rise
# logs/rough_go1/Feb09_14-57-02_ : try with 50k iterations - wooooooooow seems to climb obstacles | simulation crashed at 43050 - DEPLOYED ON THE ROBOT - CAN CLIMB STUFF
# - Feet stumble did help a lot - might even increase it
# - Robot base needs to be "stiffer" increase a bit orientation penalty
# Action smoothness 1 to -0.001
# Try with stairs up and down