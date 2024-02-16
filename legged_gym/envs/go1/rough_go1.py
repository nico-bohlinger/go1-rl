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

class GO1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        unit_obs = 45
        num_observations = 45*5 # 48
        num_privileged_obs = 45*5+241 # 45*5+241# None # 241 # 187 (height) +51 (contact forces) + 3 (base lin vel) #None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        priv_observe_contact_forces = True
        priv_observe_base_lin_vel = True

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
        control_type = 'actuator_net'          # P:position, V: velocity, T:torques. actuator_net 
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 15 #25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 2 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 #20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
        terrain_proportions = [0.0, 0.0, 0.1, 0.1, 0.8]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class normalization:
        contact_force_range = [0.0, 50.0]
        class obs_scales:
            lin_vel = 1.0 # 2.0
            ang_vel = 1.0 # 0.25
            dof_pos = 1.0 # 1.0
            dof_vel = 1.0 # 0.05
            # lin_vel = 2.0
            # ang_vel = 0.25
            # dof_pos = 1.0
            # dof_vel = 0.05
            height_measurements = 5.0
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
        rand_interval_s = 8.0
        randomize_friction = True
        friction_range = [0.05, 1.25]# [0.5, 4.5]
        randomize_base_mass = True # False
        added_mass_range = [-1., 3.] # [-1., 1.]
        randomize_base_com = True
        com_pos_range = [-0.1, 0.1]
        # randomize_motor_strength = True
        # motor_strength_range = [0.9, 1.1]
        # randomize_motor_offset = True
        # motor_offset_range = [-0.02, 0.02]
        # randomize_Kp_factor = False
        # Kp_factor_range = [0.8, 1.3]
        # randomize_Kd_factor = False
        # Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        randomize_gravity = True
        gravity_range = [-1, 1]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_lag_timesteps = True
        lag_timesteps = 6


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
            torques = -0.0002 # -0.0002
            dof_pos_limits = -10.0
            dof_pos = -0.05
            action_rate = -0.01
            orientation = -0.8 #-1.0 #-0.2 #-5.
            ang_vel_xy = -0.05 # -0.05
            base_height = -10.0 # -15.0 #-30
            feet_air_time =  1.0 #2.0 #1.0
            feet_slip = -0.04
            feet_clearance = -0.0001 # -0.001
            stumble = -12.0 # -10.0
            joint_power=-2e-5
            power_distribution=-10e-5 # -10e-5
            # action_smoothness_1 = -0.001
            # action_smoothness_2 = -0.1
            # feet_air_time =  0.01
           

        # only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        # soft_dof_vel_limit = 1.
        # soft_torque_limit = 1.
        # max_contact_force = 100. # forces above this value are penalized
            

class GO1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go1'

  