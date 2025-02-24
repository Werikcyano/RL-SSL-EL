class SSLBaseEnv(gym.Env):
    def __init__(self, field_type=1, n_robots_blue=1, n_robots_yellow=1,
                 time_step=0.025):
        super().__init__()
        self.steps = 0
        self.max_ep_length = 1200
        self.time_step = time_step
        self.fps = int(1/time_step)
        self.field = Field(field_type)
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.kick_speed_x = 5.0
        self.kick_speed_z = 0
        self.max_wheel_speed = 100
        self.max_episode_steps = 1200
        self.NORM_BOUNDS = 1.2
        self.view = None
        self.score = {'blue': 0, 'yellow': 0}

        # Observation Space
        self.observation_space = None

        # Action Space
        self.action_space = None

        # Create Simulation
        self.rsim = RCGymEnv(n_robots_blue=n_robots_blue,
                            n_robots_yellow=n_robots_yellow,
                            field_type=field_type,
                            time_step=time_step)

    def _frame_to_observations(self):
        '''Only returns the observations of the active robots
        '''
        self.observations = {}
        for i in range(self.n_robots_blue):
            self.observations[f'blue_{i}'] = self._get_robot_obs('blue', i)

        for i in range(self.n_robots_yellow):
            self.observations[f'yellow_{i}'] = self._get_robot_obs('yellow', i)

    def _calculate_reward_and_done(self):
        '''Calculate reward and done conditions

        Returns:
            reward(dict): rewards of all agents
            done(dict): done conditions of all agents
            truncated(dict): truncated conditions of all agents
        '''
        raise NotImplementedError

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        raise NotImplementedError

    def _get_commands(self, actions):
        '''Translate actions received to commands for the robots

        Args:
            actions (dict): actions of the robots

        Returns:
            commands (list): commands to be executed by the robots
        '''
        commands = []
        for i in range(self.n_robots_blue):
            robot_id = f'blue_{i}'
            if robot_id in actions:
                v_wheel0, v_wheel1, v_wheel2, v_wheel3, kick_speed_x = self._action_to_commands(
                    actions[robot_id])
                commands.append(Command(robot_id='blue', id=i, yellow=False,
                                     wheel0=v_wheel0, wheel1=v_wheel1,
                                     wheel2=v_wheel2, wheel3=v_wheel3,
                                     kick_speed_x=kick_speed_x,
                                     kick_speed_z=self.kick_speed_z))

        for i in range(self.n_robots_yellow):
            robot_id = f'yellow_{i}'
            if robot_id in actions:
                v_wheel0, v_wheel1, v_wheel2, v_wheel3, kick_speed_x = self._action_to_commands(
                    actions[robot_id])
                commands.append(Command(robot_id='yellow', id=i, yellow=True,
                                     wheel0=v_wheel0, wheel1=v_wheel1,
                                     wheel2=v_wheel2, wheel3=v_wheel3,
                                     kick_speed_x=kick_speed_x,
                                     kick_speed_z=self.kick_speed_z))

        return commands

    def _action_to_commands(self, action):
        '''Translate actions received to wheel commands

        Args:
            action (n-array): action values

        Returns:
            v_wheel0 (float): first wheel speed
            v_wheel1 (float): second wheel speed
            v_wheel2 (float): third wheel speed
            v_wheel3 (float): fourth wheel speed
            kick_speed_x (float): kick speed
        '''
        raise NotImplementedError

    def _get_robot_obs(self, team, id):
        '''Calculate robot observation

        Args:
            team (str): team that robot belongs
            id (int): robot id

        Returns:
            observation (n-array): observation array of the robot
        '''
        raise NotImplementedError

    def step(self, action):
        '''Apply action to the environment

        Args:
            action (dict): contains actions of all robots

        Returns:
            observation (dict): observation of all robots
            reward (dict): reward of all robots
            done (bool): whether episode is done
            info (dict): additional information
        '''
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
        reward, done, truncated = self._calculate_reward_and_done()

        if self.steps >= self.max_ep_length:
            done = {'__all__': False}
            truncated = {'__all__': True}

        # Check if ball went out of bounds
        ball = self.frame.ball
        half_len = self.field.length / 2
        half_wid = self.field.width / 2

        if ball.x <= -half_len or ball.x >= half_len:
            # Chama track_reset para endline
            if hasattr(self, 'track_reset'):
                self.track_reset('endline')
        elif ball.y <= -half_wid or ball.y >= half_wid:
            # Chama track_reset para lateral
            if hasattr(self, 'track_reset'):
                self.track_reset('lateral')

        infos = {
            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
        }

        if done.get("__all__", False) or truncated.get("__all__", False):
            for i in range(self.n_robots_blue):
                infos[f'blue_{i}']["score"] = self.score.copy()
                if hasattr(self, 'continuity_metrics'):
                    infos[f'blue_{i}']["continuity_metrics"] = self.continuity_metrics.copy()

            for i in range(self.n_robots_yellow):
                infos[f'yellow_{i}']["score"] = self.score.copy()
                if hasattr(self, 'continuity_metrics'):
                    infos[f'yellow_{i}']["continuity_metrics"] = self.continuity_metrics.copy()

        return self.observations.copy(), reward, done, truncated, infos

    def reset(self, *, seed=None, options=None):
        '''Reset environment

        Returns:
            observation (dict): observation of all robots
            info (dict): additional information
        '''
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        # Close render window
        del(self.view)
        self.view = None

        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        blue = {f'blue_{i}': {} for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}':{} for i in range(self.n_robots_yellow)}
        self.score = {'blue': 0, 'yellow': 0}

        self._frame_to_observations()

        return self.observations.copy(), {**blue, **yellow}

    def render(self, mode='human'):
        '''Render environment

        Args:
            mode (str): mode of rendering
        '''
        if self.view is None:
            self.view = View()

        self.view.update_frame(self.frame)

    def close(self):
        '''Close environment'''
        del(self.view)
        self.view = None 