import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree


class SSLShootEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal on a field with static defenders

        Description:

        Observation:
            Type: Box(4 + 8*n_robots_blue + 2*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized
            0->3     Ball [X, Y, V_x, V_y]
            4+i*7->10+i*7    id i(0-2) Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            24+j*5,25+j*5     id j(0-2) Yellow Robot [X, Y, v_x, v_y, v_theta]
        Actions:
            Type: Box(4, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
            3       id 0 Blue Kick x Speed  (%)

        Reward:
            1 if goal
        Starting State:
            Robot on field center, ball and defenders randomly positioned on
            positive field side
        Episode Termination:
            Goal, 25 seconds (1000 steps), or rule infraction
    """

    def __init__(self, field_type=2):
        super().__init__(field_type=field_type, n_robots_blue=3,
                         n_robots_yellow=3, time_step=0.025)

        self.max_dribbler_time = 100
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(4,), dtype=np.float32)

        n_obs = 4 + 7 * self.n_robots_blue + 5 * self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs,),
                                                dtype=np.float32)

        # scale max dist rw to 1 Considering that max possible move rw if ball and robot are in opposite corners of field
        self.ball_dist_scale = np.linalg.norm([self.field.width, self.field.length / 2])
        self.ball_grad_scale = np.linalg.norm([self.field.width / 2, self.field.length / 2]) / 4

        # scale max energy rw to 1 Considering that max possible energy if max robot wheel speed sent every step
        wheel_max_rad_s = 160
        max_steps = 1000
        self.energy_scale = ((wheel_max_rad_s * 4) * max_steps)
        self.previous_ball_potential = None

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.kick_speed_x = 5.0


        self.active_robot_idx = 0
        self.last_possession_robot_id = -1
        self.possession_robot_idx = -1
        self.last_ori_value = 0
        self.done_limit = None
        self.dribbler_time = 0
        self.commands = None
        self.reward_shaping_total = {
            'goal': 0,
            'done_left_out': 0,
            'done_ball_out': 0,
            'done_robot_out': 0,
            'done_robot_in_gk_area': 0,
            'rw_ball_grad': 0,
            'rw_robot_grad': 0,
            'rw_robot_orientation': 0,
            'rw_energy': 0
        }
        print('Environment initialized', "Obs:", n_obs)

    def reset(self):
        self.dribbler_time = 0
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)

        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))
            # observation.append(1 if self.frame.robots_blue[i].infrared else 0)

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        nearest_blue_robot, nearest_blue_robot_dist = self.get_nearest_robot_idx(
            [self.frame.ball.x, self.frame.ball.y], "blue")
        nearest_yellow_robot, nearest_yellow_robot_dist = self.get_nearest_robot_idx(
            [self.frame.ball.x, self.frame.ball.y], "yellow")

        threshold = 0.15
        self.last_possession = self.possession_robot_idx

        self.active_robot_idx = nearest_blue_robot
        self.possession_robot_idx = -1
        if self.commands is not None:
            if nearest_blue_robot_dist <= nearest_yellow_robot_dist:
                if nearest_blue_robot_dist <= threshold and self.commands[
                    nearest_blue_robot].dribbler and self.__is_toward_ball("blue", nearest_blue_robot):
                    self.possession_robot_idx = nearest_blue_robot
            else:
                if nearest_yellow_robot_dist <= threshold and self.commands[
                    self.n_robots_blue + nearest_yellow_robot].dribbler and self.__is_toward_ball("yellow",
                                                                                                  nearest_blue_robot):
                    self.possession_robot_idx = 3 + nearest_yellow_robot

            # if self.possession_robot_idx == self.last_possession and self.possession_robot_idx == self.active_robot_idx:
            #     self.dribbler_time += 1
            # else:
            #     self.dribbler_time = 0
        #
        # observation.append(self.possession_robot_idx)
        # observation.append(self.active_robot_idx)


        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        # Blue robot
        for i in range(self.n_robots_blue):
            if i != self.active_robot_idx:
                random_actions = self.action_space.sample()
                angle = self.frame.robots_blue[i].theta
                v_x, v_y, v_theta = self.convert_actions(random_actions, np.deg2rad(angle))
                cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta,
                            kick_v_x=self.kick_speed_x if random.uniform(-1, 1) < actions[3] else 0.,
                            dribbler=True)
                commands.append(cmd)
            else:
                # Controlled robot
                angle = self.frame.robots_blue[self.active_robot_idx].theta
                v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
                cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta,
                            kick_v_x=self.kick_speed_x if random.uniform(-1, 1) < actions[3] else 0.,
                            dribbler=True)
                commands.append(cmd)

        # Yellow robot
        for i in range(self.n_robots_yellow):
            random_actions = self.action_space.sample()
            angle = self.frame.robots_yellow[i].theta
            v_x, v_y, v_theta = self.convert_actions(random_actions, np.deg2rad(angle))
            cmd = Robot(yellow=True, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta,
                        kick_v_x=self.kick_speed_x if random_actions[3] > 0 else 0.,
                        dribbler=True)
            commands.append(cmd)
        self.commands = commands
        return commands

    def convert_actions(self, action, angle):

        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x * np.cos(angle) + v_y * np.sin(angle), \
                   -v_x * np.sin(angle) + v_y * np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x * c, v_y * c
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        self.reward_shaping_total = {
            'goal': 0,
            'done_left_out':0,
            'done_ball_out': 0,
            'done_robot_out': 0,
            'done_robot_in_gk_area': 0,
            'rw_ball_grad': 0,
            'rw_robot_grad': 0,
            'rw_robot_orientation': 0,
            'rw_energy': 0
        }
        reward = 0
        done = False

        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2

        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        if robot.x < self.done_limit or ball.x < self.done_limit:
            done = True
            self.reward_shaping_total['done_left_out'] += 1
            reward = -10
        if abs(robot.y) > half_wid or abs(robot.x) > half_len:
            done = True
            self.reward_shaping_total['done_robot_out'] += 1
        elif abs(ball.y) > half_wid:
            done = True
            self.reward_shaping_total['done_ball_out'] += 1
        elif ball.x < -half_len or abs(ball.y) > half_wid:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = -50
                self.reward_shaping_total['goal'] -= 1
            else:
                self.reward_shaping_total['done_ball_out'] += 1
        elif ball.x > half_len:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = 50
                self.reward_shaping_total['goal'] += 1
            else:
                self.reward_shaping_total['done_ball_out'] += 1
        elif self.last_frame is not None:

            ball_grad_rw = self.__ball_grad_rw()
            self.reward_shaping_total['rw_ball_grad'] += ball_grad_rw

            robot_grad_rw = 0.2 * self.__robot_grad_rw()
            self.reward_shaping_total['rw_robot_grad'] += robot_grad_rw

            robot_orientation_rw = 0
            if self.possession_robot_idx == self.active_robot_idx:
                robot_orientation_rw = 0.2 * self.__robot_orientation_rw()
                self.reward_shaping_total['rw_robot_orientation'] += robot_orientation_rw

            energy_rw = -self.__energy_pen() / self.energy_scale
            self.reward_shaping_total['rw_energy'] += energy_rw

            reward = ball_grad_rw + robot_grad_rw +  robot_orientation_rw + energy_rw

        done = done
        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2

        def x():
            return random.uniform(-self.field.length / 2+0.2,self.field.length / 2-0.2)

        def y():
            return random.uniform(-self.field.width/ 2+0.2,self.field.width / 2-0.2)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        def in_gk_area(obj):
            return obj.x > half_len - pen_len and abs(obj.y) < half_pen_wid

        pos_frame.ball = Ball(x=x(), y=y())
        while in_gk_area(pos_frame.ball):
            pos_frame.ball = Ball(x=x(), y=y())
        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        r = 0.09
        angle = theta()
        robot_x = pos_frame.ball.x + r * math.cos(np.deg2rad(angle))
        robot_y = pos_frame.ball.y + r * math.sin(np.deg2rad(angle))
        pos_frame.robots_blue[0] = Robot(
            x=robot_x, y=robot_y, theta=angle + 180
        )
        self.done_limit  = robot_x - 0.5
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        min_dist = 0.2
        for i in range(self.n_robots_blue):
            if i == 0:
                continue
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

        return pos_frame

    def __energy_pen(self):
        robot = self.frame.robots_blue[0]

        # Sum of abs each wheel speed sent
        energy = abs(robot.v_wheel0) \
                 + abs(robot.v_wheel1) \
                 + abs(robot.v_wheel2) \
                 + abs(robot.v_wheel3)
        return energy

    def __towards_ball_rw(self):
        theta = math.radians(self.frame.robots_blue[0].theta)
        Xr, Yr, theta, Xb, Yb = self.frame.robots_blue[0].x, self.frame.robots_blue[
            0].y, theta, self.frame.ball.x, self.frame.ball.y

        # 计算机器人-球连线方向的角度
        line_angle = math.atan2(Yb - Yr, Xb - Xr)

        # 计算摄像头方向和机器人-球连线方向之间的夹角
        angle = line_angle - theta

        # 将夹角转换为[-π,π]区间内的值
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi

        value = math.pi - abs(angle)
        normalized_value = (value - 0) / (math.pi - 0)
        if normalized_value > 0.9:
            normalized_value = 1
        return normalized_value

    def __is_toward_ball(self, team, idx):
        if team == "blue":
            Xr, Yr = self.frame.robots_blue[idx].x, self.frame.robots_blue[idx].y
            theta = math.radians(self.frame.robots_blue[idx].theta)
        if team == "yellow":
            theta = math.radians(self.frame.robots_yellow[idx].theta)
            Xr, Yr = self.frame.robots_yellow[idx].x, self.frame.robots_yellow[idx].y
        Xb, Yb = self.frame.ball.x, self.frame.ball.y

        # 计算机器人-球连线方向的角度
        line_angle = math.atan2(Yb - Yr, Xb - Xr)

        # 计算摄像头方向和机器人-球连线方向之间的夹角
        angle = line_angle - theta

        # 将夹角转换为[-π,π]区间内的值
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi

        value = math.pi - abs(angle)
        normalized_value = (value - 0) / (math.pi - 0)
        if normalized_value >= 0.9:
            return True
        else:
            return False

    def __move_to_ball_rw(self):
        assert (self.last_frame is not None)

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        last_robot = self.last_frame.robots_blue[0]
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_dist = np.linalg.norm(last_robot_pos - last_ball_pos)

        # Calculate new ball dist
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])
        dist = np.linalg.norm(robot_pos - last_ball_pos)

        move_to_ball_rw = last_dist - dist
        move_to_ball_rw = move_to_ball_rw * (1/0.05)
        return np.clip(move_to_ball_rw, -1, 1)

    def __ball_grad_rw(self):
        assert (self.last_frame is not None)

        # Goal pos
        up_goalpost = np.array([self.field.length / 2 , self.field.goal_width/2])
        down_goalpost = np.array([self.field.length / 2 , -self.field.goal_width/2])
        mid_goalpost = np.array([self.field.length / 2, 0.])

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        ball = self.frame.ball

        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = min(np.linalg.norm(up_goalpost - last_ball_pos),np.linalg.norm(down_goalpost - last_ball_pos), np.linalg.norm(mid_goalpost - last_ball_pos))

        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = min(np.linalg.norm(up_goalpost - ball_pos),np.linalg.norm(down_goalpost - ball_pos),np.linalg.norm(mid_goalpost - ball_pos))


        ball_grad = last_ball_dist - ball_dist
        ball_grad = ball_grad * (1/0.1)
        return np.clip(ball_grad, -1, 1)

    def __robot_grad_rw(self):
        assert (self.last_frame is not None)

        # Goal pos
        up_goalpost = np.array([self.field.length / 2 , self.field.goal_width/2])
        down_goalpost = np.array([self.field.length / 2 , -self.field.goal_width/2])
        mid_goalpost = np.array([self.field.length / 2, 0.])

        # Calculate previous ball dist
        last_robot = self.last_frame.robots_blue[self.active_robot_idx]
        robot = self.frame.robots_blue[self.active_robot_idx]

        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_robot_dist = min(np.linalg.norm(up_goalpost - last_robot_pos),np.linalg.norm(down_goalpost - last_robot_pos), np.linalg.norm(mid_goalpost - last_robot_pos))

        # Calculate new ball dist
        robot_pos = np.array([robot.x, robot.y])
        robot_dist = min(np.linalg.norm(up_goalpost - robot_pos),np.linalg.norm(down_goalpost - robot_pos),np.linalg.norm(mid_goalpost - robot_pos))


        ball_grad = last_robot_dist - robot_dist
        ball_grad = ball_grad * (1/0.1)
        return np.clip(ball_grad, -1, 1)

    def __robot_orientation_rw(self):
        last_ori_value = self.last_ori_value
        # 机器人坐标、朝向
        theta = math.radians(self.frame.robots_blue[0].theta)
        Xr, Yr, theta= self.frame.robots_blue[0].x, self.frame.robots_blue[
            0].y, theta

        # 球门坐标
        Xg, Yg = self.field.length / 2, 0

        # 计算机器人-球门连线方向的角度
        line_angle = math.atan2(Yg - Yr, Xg - Xr)

        # 计算摄像头方向和机器人-球门连线方向之间的夹角
        angle = line_angle - theta

        # 将夹角转换为[-π,π]区间内的值
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi

        value = math.pi - abs(angle)
        normalized_value = (value - 0) / (math.pi - 0)
        self.last_ori_value = normalized_value
        return normalized_value - last_ori_value