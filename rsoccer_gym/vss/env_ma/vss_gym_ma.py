import copy
import math
import os
import pickle
import random
from typing import Dict, List

import gym
import numpy as np
import torch
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv


class VSSMAEnv(VSSBaseEnv):
    """This environment controls N robots in a VSS soccer League 3v3 match


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot Xw
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
            -3              robot goal x
            -2              robot goal y
            -1              robot role 0=attacker 1=defender
        Actions:
            Type: Box(N, 2)
            For each blue robot in control:
                Num     Action
                0       Left Wheel Speed  (%)
                1       Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                For all robots:
                    Goal
                    Ball Potential Gradient
                Individual:
                    Move to Ball/Goal
                    Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self, n_robots_control=3):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.025)
        self.n_robots_control = n_robots_control
        self.action_space = gym.spaces.Box(low=-1,
                                           high=1,
                                           shape=(n_robots_control, 2))
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_robots_control, 43),
                                                dtype=np.float32)

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.info = {'goal_score': 0, 'ball_grad': 0,'ball_grad_opp': 0,
                     'goals_blue': 0, 'goals_yellow': 0}
        self.individual_reward = {}
        self.v_wheel_deadzone = 0.05
        self.observation = None
        self.ou_actions = []
        self.goal = [[0, 0], [0, 0]]
        self.attacker = [0, 1, 2]
        self.defender_1 = 0
        self.defender_2 = 0
        self.writer = None
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        # single attacker = 0
        # defenfer will be attacker when reach goal = 1
        # three attacker =2
        # three attacker when yellow control the ball, one attacker when blue control the ball = 3

        self.multiple_attacker_mode = 3

        # -------------------------------------Evaluation------------------------------------------------------------
        self.possession = [-1, -1]
        self.fixed_initial_position = False
        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.info = {'goal_score': 0, 'ball_grad': 0,
                     'goals_blue': 0, 'goals_yellow': 0}
        self.previous_ball_potential = None
        self.individual_reward = {}
        for ou in self.ou_actions:
            ou.reset()
        return super().reset()

    def reach_goal(self, robot_idx, team, target):
        target = np.array(target)
        if team == "blue":
            robot = np.array([self.frame.robots_blue[robot_idx].x,
                              self.frame.robots_blue[robot_idx].y])
        elif team == "yellow":
            robot = np.array([self.frame.robots_yellow[robot_idx].x,
                              self.frame.robots_yellow[robot_idx].y])
        else:
            Exception("team must be blue or yellow")
        robot_distance_to_target = np.sqrt(sum((robot - target) ** 2 for robot, target in zip(robot, target)))

        return robot_distance_to_target < 0.1

    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        intercept = 0
        # 0 no change
        # 1 blue overtake yellow
        # -1 yellow overtake blue
        passing = 0
        # 0 no pass
        # 1 blue pass
        # -1 yellow pass

        last_possession_team = copy.deepcopy(self.possession[0])
        last_possession_robot = copy.deepcopy(self.possession[1])
        new_possession_team = self.ball_possession()[0]
        new_possession_robot = self.ball_possession()[1]
        if last_possession_team != 0 and new_possession_team == 0:
            intercept = 1
        elif last_possession_team != 1 and new_possession_team == 1:
            intercept = -1

        if last_possession_team == 0 and new_possession_team == 0 and last_possession_robot != new_possession_robot:
            passing = 1
        elif last_possession_team == 1 and new_possession_team == 1 and last_possession_robot != new_possession_robot:
            passing = -1

        self.info["possession_team"] = self.possession[0]
        self.info["possession_robot_idx"] = self.possession[1]
        self.info["intercept"] = intercept
        self.info["passing"] = passing

        if self.multiple_attacker_mode == 3:
            if intercept == 1:
                self.attacker = [self.possession[1]]
                self.defender_1 = self._get_closet_robot_idx(self.goal[0], "blue", except_idx=self.attacker[0])
                for i in range(self.n_robots_control):
                    if i != self.attacker[0] and i != self.defender_1:
                        self.defender_2 = i
            elif intercept == -1:
                self.attacker = [i for i in range(self.n_robots_blue)]

        if self.multiple_attacker_mode == 1:
            if self.defender_1 not in self.attacker:
                if self.reach_goal(self.defender_1, "blue", self.goal[0]):
                    self.attacker.append(self.defender_1)
            if self.defender_2 not in self.attacker:
                if self.reach_goal(self.defender_2, "blue", self.goal[1]):
                    self.attacker.append(self.defender_2)

        for i in range(2):
            robot = Robot()
            robot.id = i
            robot.x = self.goal[i][0]
            robot.y = self.goal[i][1]
            robot.theta = 0
            self.frame.robots_blue_goal[robot.id] = robot

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        # print(f"attaker={self.attacker}, def1= {self.defender_1}, def2={self.defender_2}")

        return observation, reward, done, self.info

    def set_attacker_and_goal(self, goal):
        self.goal = goal
        if self.multiple_attacker_mode == 2:
            self.attacker = [0, 1, 2]
        elif self.multiple_attacker_mode == 3:
            pass
        else:
            self.attacker = [self._get_closet_robot_idx([self.frame.ball.x, self.frame.ball.y], "blue")]
            self.defender_1 = self._get_closet_robot_idx(goal[0], "blue", except_idx=self.attacker[0])
            for i in range(self.n_robots_control):
                if i != self.attacker[0] and i != self.defender_1:
                    self.defender_2 = i
        self._frame_to_observations()

    def get_rotated_obs(self):
        robots_dict = dict()
        for i in range(self.n_robots_blue):
            robots_dict[i] = list()
            robots_dict[i].append(self.norm_pos(self.frame.robots_blue[i].x))
            robots_dict[i].append(self.norm_pos(self.frame.robots_blue[i].y))
            robots_dict[i].append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots_dict[i].append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots_dict[i].append(self.norm_v(self.frame.robots_blue[i].v_x))
            robots_dict[i].append(self.norm_v(self.frame.robots_blue[i].v_y))
            robots_dict[i].append(self.norm_w(self.frame.robots_blue[i].v_theta))

        rotaded_obs = list()
        for i in range(self.n_robots_control):
            aux_dict = {}
            aux_dict.update(robots_dict)
            rotated = list()
            rotated = rotated + aux_dict.pop(i)
            teammates = list(aux_dict.values())
            for teammate in teammates:
                rotated = rotated + teammate
            rotaded_obs.append(rotated)

        return rotaded_obs

    def _frame_to_observations(self):

        observations = list()
        robots = self.get_rotated_obs()
        for idx in range(self.n_robots_control):
            observation = []

            observation.append(self.norm_pos(self.frame.ball.x))
            observation.append(self.norm_pos(self.frame.ball.y))
            observation.append(self.norm_v(self.frame.ball.v_x))
            observation.append(self.norm_v(self.frame.ball.v_y))

            observation += robots[idx]
            if self.n_robots_yellow != 0:
                for i in range(self.n_robots_yellow):
                    observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
                    observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
                    observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
                    observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
                    observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

            # append goal and role
            if idx in self.attacker:
                observation.append(self.norm_pos(self.frame.ball.x))
                observation.append(self.norm_pos(self.frame.ball.y))
                observation.append(0)
            elif idx == self.defender_1:
                observation.append(self.norm_pos(self.goal[0][0]))
                observation.append(self.norm_pos(self.goal[0][1]))
                observation.append(1)
            elif idx == self.defender_2:
                observation.append(self.norm_pos(self.goal[1][0]))
                observation.append(self.norm_pos(self.goal[1][1]))
                observation.append(1)
            else:
                raise Exception(
                    f"idx{idx} is neither attacker nor defender: {self.attacker},{self.defender_1},{self.defender_2}")
            observations.append(np.array(observation, dtype=np.float32))
        # Append coach observation
        observations.append(np.array(observations[0][:40], dtype=np.float32))
        observations = np.array(observations)
        self.observation = observations
        return observations

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        # Send random commands to the other robots
        for i in range(self.n_robots_control):
            self.actions[i] = actions[i]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[i])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        for i in range(self.n_robots_control, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue + i].sample()

            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0])
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):

        reward = {f'robot_{i}': 0 for i in range(self.n_robots_control)}
        done = False
        # for agent
        w_move = 0.2  # [-5,5]
        w_ball_grad = 0.8  # [-5,5]
        w_energy = 2e-6
        w_speed = 0.5  # 0 or -1
        w_goal = 50

        if len(self.individual_reward) == 0:
            for i in range(self.n_robots_control):
                self.individual_reward[f'blue_robot_{i}'] = {'move': 0, 'energy': 0, 'speed': 0}
                self.individual_reward[f'yellow_robot_{i}'] = {'move': 0, 'energy': 0, 'speed': 0}

        # Check if goal
        if self.frame.ball.x > (self.field.length / 2):
            self.info['goal_score'] += 1
            self.info['goals_blue'] += 1
            for i in range(self.n_robots_control):
                reward[f'robot_{i}'] = w_goal * 1
            done = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.info['goal_score'] -= 1
            self.info['goals_yellow'] += 1
            for i in range(self.n_robots_control):
                reward[f'robot_{i}'] = w_goal * -1
            done = True
        else:
            # if not goal
            if self.last_frame is not None:
                grad_ball_potential, closest_move, move_reward, energy_penalty, speed_penalty = 0, 0, 0, 0, 0
                # Calculate ball potential
                grad_ball_potential = self._ball_grad()
                self.info['ball_grad'] = w_ball_grad * grad_ball_potential  # noqa
                for idx in range(self.n_robots_control):
                    # Calculate Move reward
                    if w_move != 0:
                        if idx in self.attacker:
                            move_target = [self.frame.ball.x, self.frame.ball.y]
                        elif idx == self.defender_1:
                            move_target = self.goal[0]
                        elif idx == self.defender_2:
                            move_target = self.goal[1]
                        else:
                            raise Exception(f"idx{idx} is neither attacker nor defender")
                        move_reward = self._move_reward(idx, move_target)

                    # Calculate Energy penalty
                    if w_energy != 0:
                        energy_penalty = self._energy_penalty(robot_idx=idx)

                    # Calculate speed penalty
                    if w_speed != 0:
                        speed_dead_zone = 0.1
                        speed_x = self.observation[0][8 + 7 * idx]
                        speed_y = self.observation[0][9 + 7 * idx]
                        speed_abs = math.sqrt(math.pow(speed_x, 2) + math.pow(speed_y, 2))
                        speed_penalty = 0
                        if speed_abs <= speed_dead_zone:
                            speed_penalty = -1

                    rew = w_ball_grad * grad_ball_potential + \
                          w_move * move_reward + \
                          w_energy * energy_penalty + \
                          w_speed * speed_penalty

                    reward[f'robot_{idx}'] = rew
                    self.individual_reward[f'blue_robot_{idx}']['move'] = w_move * move_reward  # noqa
                    self.individual_reward[f'blue_robot_{idx}']['energy'] = w_energy * energy_penalty  # noqa
                    self.individual_reward[f'blue_robot_{idx}']['speed'] = w_speed * speed_penalty  # noqa
        return reward, done

    def write_log(self, writer, step_num):
        if writer is None:
            return
        if self.writer is None:
            self.writer = writer
        self.writer.add_scalar(f'Ball Grad Reward', self.info['ball_grad'], global_step=step_num)

        # self.writer.add_scalar(f'Opp Ball Grad Reward', self.info['opp_ball_grad'], global_step=step_num)
        for idx in range(self.n_robots_control):
            self.writer.add_scalar(f'Blue Agent_{idx} Move Reward', self.individual_reward[f'blue_robot_{idx}']['move'],
                                   global_step=step_num)
            self.writer.add_scalar(f'Blue Agent_{idx} Energy Penalty', self.individual_reward[f'blue_robot_{idx}']['energy'],
                                   global_step=step_num)
            self.writer.add_scalar(f'Blue Agent_{idx} Speed Penalty', self.individual_reward[f'blue_robot_{idx}']['speed'],
                                   global_step=step_num)
            self.writer.add_scalar(f'Yellow Agent_{idx} Move Reward', self.individual_reward[f'yellow_robot_{idx}']['move'],
                                   global_step=step_num)
            self.writer.add_scalar(f'Yellow Agent_{idx} Energy Penalty', self.individual_reward[f'yellow_robot_{idx}']['energy'],
                                   global_step=step_num)
            self.writer.add_scalar(f'Yellow Agent_{idx} Speed Penalty', self.individual_reward[f'yellow_robot_{idx}']['speed'],
                                   global_step=step_num)

    def _get_closet_robot_idx(self, target, team, except_idx=None):
        robots_distance_to_target = {}
        for idx in range(self.n_robots_control):
            target = np.array(target)
            if team == "blue":
                robot = np.array([self.frame.robots_blue[idx].x,
                                  self.frame.robots_blue[idx].y])
            elif team == "yellow":
                robot = np.array([self.frame.robots_yellow[idx].x,
                                  self.frame.robots_yellow[idx].y])
            else:
                Exception("team must be blue or yellow")
            robot_distance_to_target = np.sqrt(sum((robot - target) ** 2 for robot, target in zip(robot, target)))
            robots_distance_to_target[idx] = robot_distance_to_target
        sorted_list = sorted(robots_distance_to_target.items(), key=lambda kv: [kv[1], kv[0]])
        if except_idx is not None:
            if sorted_list[0][0] == except_idx:
                return sorted_list[1][0]
        return sorted_list[0][0]

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x():
            return random.uniform(-field_half_length + 0.1,
                                  field_half_length - 0.1)

        def y():
            return random.uniform(-field_half_width + 0.1,
                                  field_half_width - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        if self.fixed_initial_position == True:
            pos_frame.ball = Ball(x=0, y=0)

            pos_frame.robots_blue[0] = Robot(x=-0.375, y=0.4, theta=0)
            pos_frame.robots_blue[1] = Robot(x=-0.375, y=0, theta=0)
            pos_frame.robots_blue[2] = Robot(x=-0.375, y=-0.4, theta=0)

            pos_frame.robots_yellow[0] = Robot(x=0.375, y=0.4, theta=180)
            pos_frame.robots_yellow[1] = Robot(x=0.375, y=0, theta=180)
            pos_frame.robots_yellow[2] = Robot(x=0.375, y=-0.4, theta=180)

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed

    def _ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) \
                      + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def _move_reward(self, robot_idx, target, color="blue"):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        target = np.array(target)
        if color == "blue":
            robot = np.array([self.frame.robots_blue[robot_idx].x,
                              self.frame.robots_blue[robot_idx].y])
            robot_vel = np.array([self.frame.robots_blue[robot_idx].v_x,
                                  self.frame.robots_blue[robot_idx].v_y])
        else:
            robot = np.array([self.frame.robots_yellow[robot_idx].x,
                              self.frame.robots_yellow[robot_idx].y])
            robot_vel = np.array([self.frame.robots_yellow[robot_idx].v_x,
                                  self.frame.robots_yellow[robot_idx].v_y])
        robot_ball = target - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def _energy_penalty(self, robot_idx: int):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[robot_idx].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[robot_idx].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty

    # -------------------------------------Evaluation------------------------------------------------------------
    def ball_possession(self):
        '''
        return [team_id, robot_id]
        team_id:
        0: blue team
        1: yellow team
        -1: no team possesses the ball
        robot_id:
        0: id 0 robot
        1: id 1 robot
        2: id 2 robot
        -1: no robot possesses the ball
        '''
        possession = [-1, -1]
        ball_position = []
        robots_blue_position = []
        robots_yellow_position = []

        ball_position.append(self.frame.ball.x)
        ball_position.append(self.frame.ball.y)
        for i in range(self.n_robots_blue):
            robots_blue_position.append(self.frame.robots_blue[i].x)
            robots_blue_position.append(self.frame.robots_blue[i].y)
            robots_blue_position.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            robots_blue_position.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
        for i in range(self.n_robots_yellow):
            robots_yellow_position.append(self.frame.robots_yellow[i].x)
            robots_yellow_position.append(self.frame.robots_yellow[i].y)
            robots_yellow_position.append(np.sin(np.deg2rad(self.frame.robots_yellow[i].theta)))
            robots_yellow_position.append(np.cos(np.deg2rad(self.frame.robots_yellow[i].theta)))

        possession_number = 0
        for i in range(self.n_robots_blue):
            robot_distance_to_ball_edge = np.linalg.norm(
                np.array([robots_blue_position[i * 4], robots_blue_position[i * 4 + 1]]) - np.array(ball_position))
            if robot_distance_to_ball_edge <= 0.06135:
                possession = [0, i]
                possession_number = possession_number + 1
            else:
                a = np.array([robots_blue_position[i * 4], robots_blue_position[i * 4 + 1]]) + np.array(
                    [0.04 * (-robots_blue_position[i * 4 + 3] + robots_blue_position[i * 4 + 2]),
                     0.04 * (-robots_blue_position[i * 4 + 2] - robots_blue_position[i * 4 + 3])])
                b = np.array([robots_blue_position[i * 4], robots_blue_position[i * 4 + 1]]) + np.array(
                    [0.04 * (robots_blue_position[i * 4 + 3] + robots_blue_position[i * 4 + 2]),
                     0.04 * (robots_blue_position[i * 4 + 2] - robots_blue_position[i * 4 + 3])])
                c = np.array([robots_blue_position[i * 4], robots_blue_position[i * 4 + 1]]) + np.array(
                    [0.04 * (robots_blue_position[i * 4 + 3] - robots_blue_position[i * 4 + 2]),
                     0.04 * (robots_blue_position[i * 4 + 2] + robots_blue_position[i * 4 + 3])])
                d = np.array([robots_blue_position[i * 4], robots_blue_position[i * 4 + 1]]) + np.array(
                    [0.04 * (-robots_blue_position[i * 4 + 3] - robots_blue_position[i * 4 + 2]),
                     0.04 * (-robots_blue_position[i * 4 + 2] + robots_blue_position[i * 4 + 3])])
                da = np.linalg.norm(np.array(ball_position) - a)
                db = np.linalg.norm(np.array(ball_position) - b)
                dc = np.linalg.norm(np.array(ball_position) - c)
                dd = np.linalg.norm(np.array(ball_position) - d)
                # print('ball x:{}, y:{}'.format(ball_position[0],ball_position[1]))
                # print('blue i:{}, x:{}, y:{}'.format(i,robots_blue_position[i * 4], robots_blue_position[i * 4 + 1]))
                # print('blue i:{},a:{},b:{},c:{},d:{}'.format(i,a,b,c,d))
                # print('blue i:{},da:{},db:{},dc:{},dd:{}'.format(i, da, db, dc, dd))
                dn = 0.022  # 0.02135
                if da <= dn or db <= dn or dc <= dn or dd <= dn:
                    possession = [0, i]
                    possession_number = possession_number + 1
                    # print('11111111111')
        for i in range(self.n_robots_yellow):
            robot_distance_to_ball_edge = np.linalg.norm(
                np.array([robots_yellow_position[i * 4], robots_yellow_position[i * 4 + 1]]) - np.array(ball_position))
            if robot_distance_to_ball_edge <= 0.06135:
                possession = [1, i]
                possession_number = possession_number + 1
            else:
                a = np.array([robots_yellow_position[i * 4], robots_yellow_position[i * 4 + 1]]) + np.array(
                    [0.04 * (-robots_yellow_position[i * 4 + 3] + robots_yellow_position[i * 4 + 2]),
                     0.04 * (-robots_yellow_position[i * 4 + 2] - robots_yellow_position[i * 4 + 3])])
                b = np.array([robots_yellow_position[i * 4], robots_yellow_position[i * 4 + 1]]) + np.array(
                    [0.04 * (robots_yellow_position[i * 4 + 3] + robots_yellow_position[i * 4 + 2]),
                     0.04 * (robots_yellow_position[i * 4 + 2] - robots_yellow_position[i * 4 + 3])])
                c = np.array([robots_yellow_position[i * 4], robots_yellow_position[i * 4 + 1]]) + np.array(
                    [0.04 * (robots_yellow_position[i * 4 + 3] - robots_yellow_position[i * 4 + 2]),
                     0.04 * (robots_yellow_position[i * 4 + 2] + robots_yellow_position[i * 4 + 3])])
                d = np.array([robots_yellow_position[i * 4], robots_yellow_position[i * 4 + 1]]) + np.array(
                    [0.04 * (-robots_yellow_position[i * 4 + 3] - robots_yellow_position[i * 4 + 2]),
                     0.04 * (-robots_yellow_position[i * 4 + 2] + robots_yellow_position[i * 4 + 3])])
                da = np.linalg.norm(np.array(ball_position) - a)
                db = np.linalg.norm(np.array(ball_position) - b)
                dc = np.linalg.norm(np.array(ball_position) - c)
                dd = np.linalg.norm(np.array(ball_position) - d)
                dn = 0.022  # 0.02135
                if da <= dn or db <= dn or dc <= dn or dd <= dn:
                    possession = [1, i]
                    possession_number = possession_number + 1
                    # print('222222222')
        if possession_number > 1:
            possession = [-1, -1]
        if possession_number == 0 and self.possession[0] != -1:
            possession = self.possession
        self.possession = possession
        # print(possession_number)
        # print(possession)
        return possession


class VSSMAOpp(VSSMAEnv):

    def __init__(self, n_robots_control=3):
        super().__init__(n_robots_control=n_robots_control)
        self.args = None
        self.opps = []
        self.opp_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../models/opponent/maddpg/"
        self.opp_obs = None


    def load_opp(self,opp_path=None):
        if opp_path is not None:
            self.opp_path=opp_path
        print("Load opponent...")
        self.opps = []
        with open(f"{self.opp_path}args.npy", 'rb') as f:
            self.args = pickle.load(f)
        for i in range(self.n_robots_yellow):
            ckp_path = f"{self.opp_path}opp_{i}"
            agent = MATD3(self.args, i, None)
            state_dict = torch.load(ckp_path)
            agent.actor.load_state_dict(state_dict)
            agent.actor.eval()
            self.opps.append(agent)
            print(f"successfully load opp: {ckp_path}")

    def _opp_obs(self):
        observation = []
        observation.append(self.norm_pos(-self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(-self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        #  we reflect the side that the opp is attacking,
        #  so that he will attack towards the goal where the goalkeeper is
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(-self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

            observation.append(
                np.sin(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(
                -np.cos(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(self.norm_v(-self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))

            observation.append(self.norm_w(-self.frame.robots_yellow[i].v_theta))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(-self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(self.norm_v(-self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_blue[i].v_theta))

        # get rotated
        observations = []
        observations.append(observation)
        player_0 = copy.deepcopy(observation[4 + (7 * 0): 11 + (7 * 0)])
        player_1 = copy.deepcopy(observation[4 + (7 * 1): 11 + (7 * 1)])
        player_2 = copy.deepcopy(observation[4 + (7 * 2): 11 + (7 * 2)])
        observation_1 = copy.deepcopy(observation)
        observation_1[4 + (7 * 0): 11 + (7 * 0)] = player_1
        observation_1[4 + (7 * 1): 11 + (7 * 1)] = player_0
        observations.append(observation_1)
        observation_2 = copy.deepcopy(observation)
        observation_2[4 + (7 * 0): 11 + (7 * 0)] = player_2
        observation_2[4 + (7 * 2): 11 + (7 * 2)] = player_0
        observations.append(observation_2)
        observations = np.array(observations, dtype=np.float32)
        self.opp_obs = observations
        return observations

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        for i in range(self.n_robots_blue):
            self.actions[i] = actions[i]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[i])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        self.opp_obs = self._opp_obs()
        for i in range(self.n_robots_yellow):
            if len(self.opps) != 0:
                a = self.opps[i].choose_action(self.opp_obs[i], noise_std=0)
                opp_action = copy.deepcopy(a)
            else:
                opp_action = self.ou_actions[self.n_robots_blue + i].sample()[i]
                print("random.")
            v_wheel1, v_wheel0 = self._actions_to_v_wheels(opp_action)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        return commands


class VSSMASelfplay(VSSMAOpp):

    def __init__(self, n_robots_control=3):
        super().__init__(n_robots_control=n_robots_control)
        self.opp_type = 0
        self.opp_coach = None
        self.opp_path = "/home/user/football/HRL/models/selfplay_opponent/"
        self.opp_goal = [[0, 0], [0, 0]]
        self.opp_attacker = [0]
        self.opp_defender_1 = 1
        self.opp_defender_2 = 2

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        if self.multiple_attacker_mode == 1:
            if self.reach_goal(self.opp_defender_1, "yellow", self.opp_goal[0]):
                self.opp_attacker.append(self.opp_defender_1)
            if self.reach_goal(self.opp_defender_2, "yellow", self.opp_goal[1]):
                self.opp_attacker.append(self.opp_defender_2)

        return observation, reward, done, self.info

    def load_opp(self):
        try:
            super().load_opp()
        except FileNotFoundError:
            print("No opponent model found. Will use random opponent.")
        coach_path = f"{self.opp_path}coach"
        if self.opp_type == 1:
            self.opp_coach = Coach_MMOE(self.args, self.writer)
            self.opp_coach.load_model(coach_path)
        else:
            self.opp_coach = Ramdom_Coach()


    def _opp_obs(self):
        observation = []
        observation.append(self.norm_pos(-self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(-self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        #  we reflect the side that the opp is attacking,
        #  so that he will attack towards the goal where the goalkeeper is
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(-self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

            observation.append(
                np.sin(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(
                -np.cos(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(self.norm_v(-self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))

            observation.append(self.norm_w(-self.frame.robots_yellow[i].v_theta))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(-self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(self.norm_v(-self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_blue[i].v_theta))

        # get rotated
        observations = []
        observations.append(observation)
        player_0 = copy.deepcopy(observation[4 + (7 * 0): 11 + (7 * 0)])
        player_1 = copy.deepcopy(observation[4 + (7 * 1): 11 + (7 * 1)])
        player_2 = copy.deepcopy(observation[4 + (7 * 2): 11 + (7 * 2)])
        observation_1 = copy.deepcopy(observation)
        observation_1[4 + (7 * 0): 11 + (7 * 0)] = player_1
        observation_1[4 + (7 * 1): 11 + (7 * 1)] = player_0
        observations.append(observation_1)
        observation_2 = copy.deepcopy(observation)
        observation_2[4 + (7 * 0): 11 + (7 * 0)] = player_2
        observation_2[4 + (7 * 2): 11 + (7 * 2)] = player_0
        observations.append(observation_2)

        for idx in range(self.n_robots_yellow):
            # append goal and role
            if idx in self.opp_attacker:
                observations[idx].append(self.norm_pos(self.frame.ball.x))
                observations[idx].append(self.norm_pos(self.frame.ball.y))
                observations[idx].append(0)
            elif idx == self.opp_defender_1:
                observations[idx].append(self.norm_pos(self.opp_goal[0][0]))
                observations[idx].append(self.norm_pos(self.opp_goal[0][1]))
                observations[idx].append(1)
            elif idx == self.opp_defender_2:
                observations[idx].append(self.norm_pos(self.opp_goal[1][0]))
                observations[idx].append(self.norm_pos(self.opp_goal[1][1]))
                observations[idx].append(1)
            else:
                raise Exception(f"idx{idx} is neither attacker nor defender")

        # Append coach observation
        observations.append(np.array(observations[0][:40], dtype=np.float32))
        observations = np.array(observations)
        self.opp_obs = observations
        return observations

    def set_attacker_and_goal(self, goal):
        super().set_attacker_and_goal(goal)
        if self.opp_obs is None:
            self._opp_obs()
        coach_obs = self.opp_obs[-1]
        opp_goal = self.opp_coach.choose_action(coach_obs)
        self.opp_goal = opp_goal
        self.opp_attacker = [self._get_closet_robot_idx([self.frame.ball.x, self.frame.ball.y], "yellow")]
        self.opp_defender_1 = self._get_closet_robot_idx(goal[0], "yellow", except_idx=self.opp_attacker[0])
        for i in range(self.n_robots_control):
            if i != self.opp_attacker[0] and i != self.opp_defender_1:
                self.opp_defender_2 = i


class VSSMAAdv(VSSMAEnv):
    def __init__(self, n_robots_control=3):
        super().__init__(n_robots_control=n_robots_control)
        self.args = None
        self.opps = []
        self.opp_obs = None
        self.opp_action = None
        self.noise = 0
    def reset(self):
        obs = super().reset()
        self._opp_obs()
        return obs

    def set_opp(self, agents,noise):
        self.opps = agents
        self.noise = noise

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        goal_blue = False
        if "goals_blue" in self.info.keys():
            if self.info["goals_blue"] == 1:
                goal_blue = True
        self.info["opp_agent_r_n"] = list(self._calculate_opp_reward(done, goal_blue).values())
        return observation, reward, done, self.info

    def _opp_obs(self):
        observation = []
        observation.append(self.norm_pos(-self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(-self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        #  we reflect the side that the opp is attacking,
        #  so that he will attack towards the goal where the goalkeeper is
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(-self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

            observation.append(
                np.sin(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(
                -np.cos(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(self.norm_v(-self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))

            observation.append(self.norm_w(-self.frame.robots_yellow[i].v_theta))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(-self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(self.norm_v(-self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_blue[i].v_theta))

        # get rotated
        observations = []
        observations.append(observation)
        player_0 = copy.deepcopy(observation[4 + (7 * 0): 11 + (7 * 0)])
        player_1 = copy.deepcopy(observation[4 + (7 * 1): 11 + (7 * 1)])
        player_2 = copy.deepcopy(observation[4 + (7 * 2): 11 + (7 * 2)])
        observation_1 = copy.deepcopy(observation)
        observation_1[4 + (7 * 0): 11 + (7 * 0)] = player_1
        observation_1[4 + (7 * 1): 11 + (7 * 1)] = player_0
        observations.append(observation_1)
        observation_2 = copy.deepcopy(observation)
        observation_2[4 + (7 * 0): 11 + (7 * 0)] = player_2
        observation_2[4 + (7 * 2): 11 + (7 * 2)] = player_0
        observations.append(observation_2)
        observations = np.array(observations, dtype=np.float32)
        self.opp_obs = observations
        return observations

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        for i in range(self.n_robots_blue):
            self.actions[i] = actions[i]
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[i])
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        self.opp_obs = self._opp_obs()
        opp_actions = []
        for i in range(self.n_robots_yellow):
            if len(self.opps) != 0:
                a = self.opps[i].choose_action(self.opp_obs[i], noise_std=self.noise)
                opp_action = copy.deepcopy(a)
            else:
                opp_action = self.ou_actions[self.n_robots_blue + i].sample()[i]
            opp_actions.append(opp_action)
            v_wheel1, v_wheel0 = self._actions_to_v_wheels(opp_action)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        self.info["opp_a_n"] = opp_actions
        return commands

    def _calculate_opp_reward(self, done, blue_goal=False):

        reward = {f'robot_{i}': 0 for i in range(self.n_robots_control)}
        # for agent
        w_move = 0.2  # [-5,5]
        w_ball_grad = 0.8  # [-5,5]
        w_energy = 2e-6
        w_speed = 0.5  # 0 or -1
        w_goal = 50

        if done:
            if blue_goal:
                for i in range(self.n_robots_control):
                    reward[f'robot_{i}'] = w_goal * -1
            else:
                for i in range(self.n_robots_control):
                    reward[f'robot_{i}'] = w_goal * 1
        else:
            # if not goal
            if self.last_frame is not None:
                grad_ball_potential, closest_move, move_reward, energy_penalty, speed_penalty = 0, 0, 0, 0, 0
                # Calculate ball potential
                weighted_grad_ball_potential = -self.info['ball_grad']
                self.info['opp_ball_grad'] = weighted_grad_ball_potential
                for idx in range(self.n_robots_control):
                    # Calculate Move reward
                    if w_move != 0:
                        move_target = [self.frame.ball.x, self.frame.ball.y]
                        move_reward = self._move_reward(idx, move_target, color="yellow")

                    # Calculate Energy penalty
                    if w_energy != 0:
                        energy_penalty = self._energy_penalty(robot_idx=idx)

                    # Calculate speed penalty
                    if w_speed != 0:
                        speed_dead_zone = 0.1
                        speed_x = self.observation[0][27 + (5 * idx)]
                        speed_y = self.observation[0][28 + (5 * idx)]
                        speed_abs = math.sqrt(math.pow(speed_x, 2) + math.pow(speed_y, 2))
                        speed_penalty = 0
                        if speed_abs <= speed_dead_zone:
                            speed_penalty = -1

                    rew = weighted_grad_ball_potential + \
                          w_move * move_reward + \
                          w_energy * energy_penalty + \
                          w_speed * speed_penalty

                    reward[f'robot_{idx}'] = rew
                    self.individual_reward[f'yellow_robot_{idx}']['move'] = w_move * move_reward  # noqa
                    self.individual_reward[f'yellow_robot_{idx}']['energy'] = w_energy * energy_penalty  # noqa
                    self.individual_reward[f'yellow_robot_{idx}']['speed'] = w_speed * speed_penalty  # noqa

        return reward


if __name__ == '__main__':
    env = VSSMAEnv()
    print(env.n_robots_yellow)
