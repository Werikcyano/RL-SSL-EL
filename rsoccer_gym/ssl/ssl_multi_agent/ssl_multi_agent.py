import numpy as np
from gymnasium.spaces import Box, Dict
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
import random
from collections import namedtuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import OrderedDict
from rsoccer_gym.Entities.Robot import Robot
import copy
from gymnasium.wrappers import RecordVideo
import random

GOAL_REWARD = 10
OUTSIDE_REWARD = -10

class SSLMultiAgentEnv(SSLBaseEnv, MultiAgentEnv):
    default_players = 3
    def __init__(self,
        init_pos,
        field_type=2, 
        fps=40,
        match_time=40,
        stack_observation=8,
        render_mode='human'
    ):
        # Métricas de continuidade
        self.continuity_metrics = {
            'total_resets': 0,
            'lateral_resets': 0,
            'endline_resets': 0,
            'time_between_resets': [],
            'last_reset_time': None,
            'current_sequence_time': 0,
            'max_sequence_without_reset': 0,
            'goals_blue': 0,
            'goals_yellow': 0,
            'total_goals': 0,
            'goals_per_episode': 0
        }

        # Garante que init_pos tenha as chaves necessárias
        if not isinstance(init_pos, dict):
            init_pos = {"blue": {}, "yellow": {}, "ball": [0, 0]}
        if "blue" not in init_pos:
            init_pos["blue"] = {}
        if "yellow" not in init_pos:
            init_pos["yellow"] = {}
        if "ball" not in init_pos:
            init_pos["ball"] = [0, 0]

        self.n_robots_blue = len(init_pos["blue"])
        self.n_robots_yellow = len(init_pos["yellow"])
        self.score = {'blue': 0, 'yellow': 0}
        self.render_mode = render_mode
        super().__init__(
            field_type=field_type, 
            n_robots_blue=self.n_robots_blue,
            n_robots_yellow=self.n_robots_yellow, 
            time_step=1/fps, 
            max_ep_length=int(match_time*fps),
            render_mode=render_mode
        )

        agent_ids_blue = [f'blue_{i}'for i in range(self.n_robots_blue)]
        agent_ids_yellow = [f'yellow_{i}'for i in range(self.n_robots_yellow)]
        self._agent_ids = [*agent_ids_blue, *agent_ids_yellow]
        self.max_ep_length = int(match_time*fps)
        self.fps = fps
        self.last_actions = {
            **{f'blue_{i}': np.zeros(4) for i in range(self.n_robots_blue)}, 
            **{f'yellow_{i}': np.zeros(4) for i in range(self.n_robots_yellow)}
        }

        self.stack_observation = stack_observation
        # Limit robot speeds
        self.max_v = 1.5
        self.max_w = 10
        self.kick_speed_x = 3.0

        self.init_pos = init_pos

        self.obs_size = 77 #obs[f'blue_0'].shape[0]
        self.act_size = 4

        self.actions_bound = {"low": -1, "high": 1}

        blue = {f'blue_{i}': Box(low=self.actions_bound["low"], high=self.actions_bound["high"], shape=(self.act_size, ), dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=self.actions_bound["low"], high=self.actions_bound["high"], shape=(self.act_size, ), dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.action_space =  Dict(**blue, **yellow)

        blue = {f'blue_{i}': Box(low=-self.NORM_BOUNDS - 0.001, high=self.NORM_BOUNDS + 0.001, shape=(self.stack_observation * self.obs_size, ), dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=-self.NORM_BOUNDS - 0.001, high=self.NORM_BOUNDS + 0.001, shape=(self.stack_observation * self.obs_size, ), dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.observation_space = Dict(**blue, **yellow)

        self.observations = {
            **{f'blue_{i}': np.zeros(self.stack_observation * self.obs_size, dtype=np.float64) for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(self.stack_observation * self.obs_size, dtype=np.float64) for i in range(self.n_robots_yellow)}
        }

    def _get_commands(self, actions):
        commands = []
        for i in range(self.n_robots_blue):
            robot_actions = actions[f'blue_{i}'].copy()
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, np.deg2rad(angle))
            cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)
        
        for i in range(self.n_robots_yellow):
            # Se estiver na Tarefa 1, mantém o robô amarelo estático
            if hasattr(self, 'task_level') and self.task_level == 1:
                cmd = Robot(yellow=True, id=i, v_x=0, v_y=0, v_theta=0, kick_v_x=0)
            else:
                robot_actions = actions[f'yellow_{i}'].copy()
                angle = self.frame.robots_yellow[i].theta
                v_x, v_y, v_theta = self.convert_actions(robot_actions, np.deg2rad(angle))
                cmd = Robot(yellow=True, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)

        return commands
    
    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""

        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta


    def _calculate_reward_done(self):
        done = {'__all__': False}
        truncated = {'__all__': False}
        ball = self.frame.ball
        last_ball = self.last_frame.ball

        Goal = namedtuple('goal', ['x', 'y'])
        blue_rw_dict = {}
        yellow_rw_dict = {}

        if self.n_robots_blue > 0:
            blue_rw = np.zeros(self.n_robots_blue)
            r_dist = -2.5
            for idx in range(self.n_robots_blue):
                blue_robot = self.frame.robots_blue[idx]
                goal_adv = Goal(x=0.2+self.field.length/2, y=0)
                goal_ally = Goal(x=-0.2-self.field.length/2, y=0)

                r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
                r_dist = max(-self._get_dist_between(ball, blue_robot), r_dist)
                r_off = self._get_3dots_angle_between(blue_robot, ball, goal_adv)[2] - 1
                r_def = self._get_3dots_angle_between(goal_ally, blue_robot, ball)[2] - 1

                blue_rw[idx] = 0.7*r_speed + 0.1*r_off + 0.1*r_def

            blue_rw += 0.1*r_dist*np.ones(self.n_robots_blue)
            blue_rw_dict = {f'blue_{id}':rw for id, rw in enumerate(blue_rw)}

        if self.n_robots_yellow > 0:
            yellow_rw = np.zeros(self.n_robots_yellow)
            r_dist = -2.5
            for idx in range(self.n_robots_yellow):
                yellow_robot = self.frame.robots_yellow[idx]
                goal_adv = Goal(x=-0.2-self.field.length/2, y=0)
                goal_ally = Goal(x=0.2+self.field.length/2, y=0)

                r_speed = self.__ball_grad_rw(ball, last_ball, goal_adv)
                r_dist = max(-self._get_dist_between(ball, yellow_robot), r_dist)
                r_off = self._get_3dots_angle_between(yellow_robot, ball, goal_adv)[2] - 1
                r_def = self._get_3dots_angle_between(goal_ally, yellow_robot, ball)[2] - 1
            
                yellow_rw[idx] = 0.7*r_speed + 0.1*r_off + 0.1*r_def

            yellow_rw += 0.1*r_dist*np.ones(self.n_robots_yellow)
            yellow_rw_dict = {f'yellow_{id}':rw for id, rw in enumerate(yellow_rw)}

        half_len = self.field.length/2 
        half_wid = self.field.width/2
        half_goal_wid = self.field.goal_width / 2

        if ball.x >= half_len and abs(ball.y) < half_goal_wid:
            done = {'__all__': True}
            self.score['blue'] += 1
            
            # Atualiza métricas de gols
            self.continuity_metrics['goals_blue'] += 1
            self.continuity_metrics['total_goals'] += 1
            self.continuity_metrics['goals_per_episode'] += 1

            blue_rw_dict = {f'blue_{i}': GOAL_REWARD for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{i}': -GOAL_REWARD for i in range(self.n_robots_yellow)}
        
        elif ball.x <= -half_len and abs(ball.y) < half_goal_wid:
            done = {'__all__': True}
            self.score['yellow'] += 1
            
            # Atualiza métricas de gols
            self.continuity_metrics['goals_yellow'] += 1
            self.continuity_metrics['total_goals'] += 1
            self.continuity_metrics['goals_per_episode'] += 1

            blue_rw_dict = {f'blue_{i}': -GOAL_REWARD for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{i}': GOAL_REWARD for i in range(self.n_robots_yellow)}
        
        elif ball.x <= -half_len or ball.x >= half_len:
            blue_rw_dict = {f'blue_{i}': OUTSIDE_REWARD for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{i}': OUTSIDE_REWARD for i in range(self.n_robots_yellow)}

            # Chama track_reset para endline
            if hasattr(self, 'track_reset'):
                self.track_reset('endline')

            initial_pos_frame: Frame = self._get_initial_positions_frame(42)
            self.rsim.reset(initial_pos_frame)
            self.frame = self.rsim.get_frame()
        
        elif (ball.y <= -half_wid or ball.y >= half_wid):
            blue_rw_dict = {f'blue_{i}': OUTSIDE_REWARD for i in range(self.n_robots_blue)}
            yellow_rw_dict = {f'yellow_{i}': OUTSIDE_REWARD for i in range(self.n_robots_yellow)}

            # Chama track_reset para lateral
            if hasattr(self, 'track_reset'):
                self.track_reset('lateral')

            initial_pos_frame: Frame = self._get_initial_positions_frame(42)
            self.rsim.reset(initial_pos_frame)
            self.frame = self.rsim.get_frame()

        reward = {**blue_rw_dict, **yellow_rw_dict}
        
        return reward, done, truncated
    
    def __ball_grad_rw(self, ball, last_ball, goal):
        assert(self.last_frame is not None)
        
        goal = np.array([goal.x, goal.y])
        # Calculate previous ball dist
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(goal - last_ball_pos)
        
        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal - ball_pos)
        diff = last_ball_dist - ball_dist
        ball_dist = min(diff - 0.01, self.kick_speed_x*(1/self.fps))
        ball_speed_rw = ball_dist/(1/self.fps)
        
        return np.clip(ball_speed_rw/self.kick_speed_x, -1, 1)

    def reset(self, seed=42, options={}):
        # Reseta métricas de continuidade
        self.continuity_metrics.update({
            'total_resets': 0,
            'lateral_resets': 0,
            'endline_resets': 0,
            'time_between_resets': [],
            'last_reset_time': None,
            'current_sequence_time': 0,
            'max_sequence_without_reset': 0,
            'goals_blue': 0,
            'goals_yellow': 0,
            'total_goals': 0,
            'goals_per_episode': 0
        })

        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        # Close render window
        del(self.view)
        self.view = None

        initial_pos_frame: Frame = self._get_initial_positions_frame(seed)
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        blue = {f'blue_{i}': {} for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}':{} for i in range(self.n_robots_yellow)}
        self.score = {'blue': 0, 'yellow': 0}

        self._frame_to_observations()

        return self.observations.copy(), {**blue, **yellow}
  
    def _get_initial_positions_frame(self, seed):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        places = KDTree()
        pos_frame: Frame = Frame()

        # Posição da bola
        if isinstance(self.init_pos["ball"], list): 
            pos_frame.ball = Ball(x=self.init_pos["ball"][0], y=self.init_pos["ball"][1])
        else:
            pos_frame.ball = Ball(x=0, y=0)  # Posição padrão no centro
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        min_dist = 0.2
        # Posições dos robôs azuis
        for i in range(self.n_robots_blue):
            key = str(i + 1)
            if key in self.init_pos['blue']:
                pos = self.init_pos['blue'][key]
                pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])
                places.insert((pos[0], pos[1]))
            else:
                # Se não houver posição definida, usa posição padrão
                x = -0.5 - i
                y = 0.0
                theta = 0.0
                pos_frame.robots_blue[i] = Robot(x=x, y=y, theta=theta)
                places.insert((x, y))

        # Posições dos robôs amarelos
        for i in range(self.n_robots_yellow):
            key = str(i + 1)
            if key in self.init_pos['yellow']:
                pos = self.init_pos['yellow'][key]
                pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])
                places.insert((pos[0], pos[1]))
            else:
                # Se não houver posição definida, usa posição padrão
                x = 1.5 + i
                y = 0.0
                theta = 180.0
                pos_frame.robots_yellow[i] = Robot(x=x, y=y, theta=theta)
                places.insert((x, y))

        return pos_frame

    def _get_pos(self, obj):

        x = self.norm_pos(obj.x)
        y = self.norm_pos(obj.y)
        v_x = self.norm_v(obj.v_x)
        v_y = self.norm_v(obj.v_y)
        
        theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
        sin = np.sin(theta) if theta is not None else None
        cos = np.cos(theta) if theta is not None else None
        theta = np.arctan2(sin, cos)/np.pi if theta is not None else None
        v_theta = self.norm_w(obj.v_theta) if theta is not None else None
        #tan = np.tan(theta) if theta else None

        return x, y, v_x, v_y, sin, cos, theta, v_theta

    def _get_3dots_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        f = lambda x: np.isnan(x).any() or np.isinf(x).any()
        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])
        p3 = np.array([obj3.x, obj3.y])

        vec1 = p1 - p2
        vec2 = p3 - p2

        if (f(p1) or f(p2) or f(p3) or f(vec1) or f(vec2)):
            with open('/ws/videos/error_arccos.txt', "w") as file:
                file.write(f"p1: {p1}\np2{p2}\np3 {p3}\nvec1 {vec1}\nvec2: {vec2}")

        cos_theta = np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return np.sin(theta), np.cos(theta), theta/np.pi

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta/np.pi
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.field.length, self.field.width])
        dist = np.linalg.norm(diff_vec)

        return np.clip(dist / max_dist, 0, 1)

    def inverted_robot(self, robot):
        return Robot(
            x=-robot.x, 
            y=robot.y, 
            theta= 180 - robot.theta if robot.theta < 180 else 540 - robot.theta, 
            v_x=-robot.v_x, 
            v_y=-robot.v_y, 
            v_theta=-robot.v_theta
        )

    def _frame_to_observations(self):

        # print("=====================================OBSERVATION===================================================")
        f = lambda x: " ".join([f"{i:.2f}" for i in x])
        ball_template = namedtuple('ball', ['x', 'y', 'v_x', 'v_y'])
        goal = namedtuple('goal', ['x', 'y'])
        for i in range(self.n_robots_blue):
            robot = self.frame.robots_blue[i] 
            robot_action = self.last_actions[f'blue_{i}']
            allys = [self.frame.robots_blue[j] for j in range(self.n_robots_blue) if j != i]
            allys_actions = [self.last_actions[f'blue_{j}'] for j in range(self.n_robots_blue) if j != i]
            advs = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow)]

            ball = ball_template(x=self.frame.ball.x, y=self.frame.ball.y, v_x=self.frame.ball.v_x, v_y=self.frame.ball.v_y)

            goal_adv = goal(x=   0.2 + self.field.length/2, y=0)
            goal_ally = goal(x= -0.2 - self.field.length/2, y=0)

            robot_obs = self.robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)
            self.observations[f'blue_{i}'] = np.delete(self.observations[f'blue_{i}'], range(len(robot_obs)))
            self.observations[f'blue_{i}'] = np.concatenate([self.observations[f'blue_{i}'], robot_obs], axis=0, dtype=np.float64)

            # if i == 1:
            #     print(f"blue_{i}")
            #     print(f"\tX={robot.x}\tY={robot.y}\tTheta={robot.theta}\tVx={robot.v_x}\tVy={robot.v_y}\tVtheta={robot.v_theta}")
            #     print(f"\t pos: {f(robot_obs[:14])} \n\t ori: {f(robot_obs[14:32])} \n\t dist: {f(robot_obs[32:40])} \n\t ang: {f(robot_obs[40:64])} \n\t last_act: {f(robot_obs[64:76])} \n\t time_left: {robot_obs[76]}")

        for i in range(self.n_robots_yellow):
            robot = self.inverted_robot(self.frame.robots_yellow[i])
            robot_action = self.last_actions[f'yellow_{i}']
            allys = [self.inverted_robot(self.frame.robots_yellow[j]) for j in range(self.n_robots_yellow) if j != i]
            allys_actions = [self.last_actions[f'yellow_{j}'] for j in range(self.n_robots_yellow) if j != i]
            advs = [self.inverted_robot(self.frame.robots_blue[j]) for j in range(self.n_robots_blue)]

            ball = ball_template(x=-self.frame.ball.x, y=self.frame.ball.y, v_x=-self.frame.ball.v_x, v_y=self.frame.ball.v_y)

            goal_adv = goal(x=  -(-0.2 - self.field.length/2), y=0)
            goal_ally = goal(x= -( 0.2 + self.field.length/2), y=0)
            
            robot_obs = self.robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)

            self.observations[f'yellow_{i}'] = np.delete(self.observations[f'yellow_{i}'], range(len(robot_obs)))
            self.observations[f'yellow_{i}'] = np.concatenate([self.observations[f'yellow_{i}'], robot_obs], axis=0, dtype=np.float64)

            # if i == 1:
            #     print(f"\nyellow_{i}")
            #     print(f"\tX={robot.x}\tY={robot.y}\tTheta={robot.theta}\tVx={robot.v_x}\tVy={robot.v_y}\tVtheta={robot.v_theta}")
            #     print(f"\t pos: {f(robot_obs[:14])} \n\t ori: {f(robot_obs[14:32])} \n\t dist: {f(robot_obs[32:40])} \n\t ang: {f(robot_obs[40:64])} \n\t last_act: {f(robot_obs[64:76])} \n\t time_left: {robot_obs[76]}")

    def robot_observation(self, robot, allys, adversaries, robot_action, allys_actions, ball, goal_adv, goal_ally):

        positions = []
        orientations = []
        dists = []
        angles = []
        last_actions = np.array([robot_action] + allys_actions).flatten()

        x_b, y_b, *_ = self._get_pos(ball)
        sin_BG_al, cos_BG_al, theta_BG_al = self._get_2dots_angle_between(goal_ally, ball)
        sin_BG_ad, cos_BG_ad, theta_BG_ad = self._get_2dots_angle_between(goal_adv, ball)
        dist_BG_al = self._get_dist_between(ball, goal_ally)
        dist_BG_ad = self._get_dist_between(ball, goal_adv)

        x_r, y_r, *_, sin_r, cos_r, theta_r, _  = self._get_pos(robot)
        sin_BR, cos_BR, theta_BR = self._get_2dots_angle_between(ball, robot)
        dist_BR = self._get_dist_between(ball, robot)

        positions.append([x_r, y_r])
        orientations.append([sin_r, cos_r, theta_r])
        dists.append([dist_BR, dist_BG_al, dist_BG_ad])
        angles.append([
            sin_BR, cos_BR, theta_BR, 
            sin_BG_al, cos_BG_al, theta_BG_al, 
            sin_BG_ad, cos_BG_ad, theta_BG_ad
        ])

        for ally in allys:
            x_al, y_al, *_, sin_al, cos_al, theta_al, _ = self._get_pos(ally)
            sin_AlR, cos_AlR, theta_AlR = self._get_2dots_angle_between(ally, robot)
            ally_dist = self._get_dist_between(ally, robot)
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])
        
        for i in range(self.default_players - len(allys) - 1):
            x_al, y_al, sin_al, cos_al, theta_al = 0, 0, 0, 0, 0
            sin_AlR, cos_AlR, theta_AlR = 0, 0, 0
            ally_dist = 0
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])

        
        for adv in adversaries:
            x_adv, y_adv, *_,  sin_adv, cos_adv, theta_adv, _ = self._get_pos(adv)
            sin_AdR, cos_AdR, theta_AdR = self._get_2dots_angle_between(adv, robot)
            adv_dist = self._get_dist_between(adv, robot)
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        for i in range(self.default_players - len(adversaries)):
            x_adv, y_adv, sin_adv, cos_adv, theta_adv = 0, 0, 0, 0, 0
            sin_AdR, cos_AdR, theta_AdR = 0, 0, 0
            adv_dist = 0
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        positions.append([x_b, y_b])

        positions = np.concatenate(positions)
        orientations = np.concatenate(orientations)
        dists = np.concatenate(dists)
        angles = np.concatenate(angles)
        time_left = [(self.max_ep_length - self.steps)/self.max_ep_length]

        #print(f"len_pos: {len(positions)} \t len_ori: {len(orientations)} \t len_dist: {len(dists)} \t len_ang: {len(angles)} \t len_last_act: {len(last_actions)} \t len_time_left: {len(time_left)}")
        robot_obs = np.concatenate([positions, orientations, dists, angles, last_actions, time_left], dtype=np.float64)
        return robot_obs
    
    def track_reset(self, reset_type: str):
        """
        Atualiza as métricas quando ocorre um reset
        Args:
            reset_type: 'lateral' ou 'endline'
        """
        current_time = self.steps / self.fps
        
        # Atualiza contadores de reset
        self.continuity_metrics['total_resets'] += 1
        self.continuity_metrics[f'{reset_type}_resets'] += 1
        
        # Calcula e registra tempo entre resets
        if self.continuity_metrics['last_reset_time'] is not None:
            time_between = current_time - self.continuity_metrics['last_reset_time']
            self.continuity_metrics['time_between_resets'].append(time_between)
            
            # Atualiza sequência máxima sem reset
            if time_between > self.continuity_metrics['max_sequence_without_reset']:
                self.continuity_metrics['max_sequence_without_reset'] = time_between
        
        self.continuity_metrics['last_reset_time'] = current_time
        self.continuity_metrics['current_sequence_time'] = 0

    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        self.last_actions = action.copy()

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        self._frame_to_observations()
        reward, done, truncated = self._calculate_reward_done()

        if self.steps >= self.max_ep_length:
            done = {'__all__': False}
            truncated = {'__all__': True}

        # Atualiza tempo da sequência atual
        current_time = self.steps / self.fps
        if self.continuity_metrics['last_reset_time'] is None:
            self.continuity_metrics['current_sequence_time'] = current_time
        else:
            self.continuity_metrics['current_sequence_time'] = current_time - self.continuity_metrics['last_reset_time']

        infos = {
            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
        }  

        if done.get("__all__", False) or truncated.get("__all__", False):
            for i in range(self.n_robots_blue):
                infos[f'blue_{i}']["score"] = self.score.copy()    
                infos[f'blue_{i}']["continuity_metrics"] = self.continuity_metrics.copy()

            for i in range(self.n_robots_yellow):
                infos[f'yellow_{i}']["score"] = self.score.copy()
                infos[f'yellow_{i}']["continuity_metrics"] = self.continuity_metrics.copy()
        
        return self.observations.copy(), reward, done, truncated, infos

class SSLMultiAgentEnv_record(RecordVideo, MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_ids = self.env._agent_ids
