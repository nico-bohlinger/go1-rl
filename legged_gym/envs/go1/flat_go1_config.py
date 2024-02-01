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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO1FlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45 # 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class viewer:
        ref_env = 0
        # pos = [10, 0, 6]  # [m]
        # lookat = [11., 5, 3.]  # [m]
        pos = [1.0, -3.5, 2]  # [m]
        lookat = [1.0, 0.5, 1.]  # [m]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'          # P:position, V: velocity, T:torques
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class normalization:
        class obs_scales:
            lin_vel = 1.0 # 2.0
            ang_vel = 1.0 # 0.25
            dof_pos = 1.0 # 1.0
            dof_vel = 1.0 # 0.05
            # height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5 # 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.05, 4.5]# [0.5, 1.25]
        randomize_base_mass = True # False
        added_mass_range = [-1., 3.] # [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.34
        class scales( LeggedRobotCfg.rewards.scales ):
            # Same as https://github.com/Anchoret13/legged_gym/blob/master/legged_gym/envs/go1/go1_config.py
            torques = -0.0002 # -0.0002
            dof_pos_limits = -10.0
            action_rate = -0.01
            orientation = -5.
            base_height = -30. #-30
        # ------------Others parameters inherent from legged_robot_config.py-----------
        #     termination = -0.0
        #     tracking_lin_vel = 1.0
        #     tracking_ang_vel = 0.5
        #     lin_vel_z = -2.0
        #     ang_vel_xy = -0.05
        #     dof_vel = -0.
        #     dof_acc = -2.5e-7
        #     feet_air_time =  1.0
        #     collision = -1.
        #     feet_stumble = -0.0 
        #     stand_still = -0.

        # only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # soft_dof_vel_limit = 1.
        # soft_torque_limit = 1.
        # base_height_target = 1.
        # max_contact_force = 100. # forces above this value are penalized
            
            

class GO1FlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 1500 # number of policy updates
        run_name = ''
        experiment_name = 'flat_go1'

  