import gym
from gym.utils import seeding
from gym.envs.classic_control import rendering


class Maze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 0.5
    }

    def __init__(self):
        self.state = None
        self.screen_width = 400  # 屏幕边长
        self.grid_width = 120  # 格子边长
        self.exploer_width = 100  # 探索者边长
        self.treasure_radius = 50  # 宝藏半径

        self.action_space = ['left', 'right', 'up', 'down']
        self.states_space = []
        for i in range(9):
            self.states_space.append(i)

        self.reward_space = dict()
        self.reward_space[0] = 0
        self.reward_space[1] = 0
        self.reward_space[2] = 1
        self.reward_space[3] = 0
        self.reward_space[4] = -1
        self.reward_space[5] = -1
        self.reward_space[6] = 0
        self.reward_space[7] = 0
        self.reward_space[8] = 0

        self.viewer = None

        self.pos = dict()
        self.pos[0] = (30, 270)
        self.pos[1] = (150, 270)
        self.pos[2] = (270, 270)
        self.pos[3] = (30, 150)
        self.pos[4] = (150, 150)
        self.pos[5] = (270, 150)
        self.pos[6] = (30, 30)
        self.pos[7] = (150, 30)
        self.pos[8] = (270, 30)

    def getActions(self):
        return self.action_space

    def getStates(self):
        return self.states_space

    def getRewards(self):
        return self.reward_space

    def _step(self, action):
        # 判断当前系统状态
        state = self.state
        if state is None:
            state = 0
        elif state == 4 or state == 5 or state == -1:
            return state, -1, True, {}
        elif state == 2:
            return state, 1, True, {}

        # 状态转移
        next_state = None
        if action in self.action_space:
            # 移动
            if (state == 0 or state == 1 or state == 2) and action == 'up':
                next_state = state
            elif (state == 0 or state == 3 or state == 6) and action == 'left':
                next_state = state
            elif (state == 6 or state == 7 or state == 8) and action == 'down':
                next_state = state
            elif (state == 2 or state == 5 or state == 8) and action == 'right':
                next_state = state
            elif action == 'up':
                next_state = state - 3
            elif action == 'left':
                next_state = state - 1
            elif action == 'down':
                next_state = state + 3
            elif action == 'right':
                next_state = state + 1

        state = next_state
        self.state = state

        done = False
        if self.reward_space[state] == -1 or self.reward_space[state] == 1:
            done = True
        r = self.reward_space[state]
        return state, r, done, {}

    def _reset(self):
        self.state = 8
        return self.state

    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_width)

            # 画线
            self.lines = []
            # 横线
            for i in range(4):
                line = rendering.Line((20, 20 + self.grid_width * i), (380, 20 + self.grid_width * i))
                self.lines.append(line)
            # 纵线
            for i in range(4):
                line = rendering.Line((20 + self.grid_width * i, 20), (20 + self.grid_width * i, 380))
                self.lines.append(line)
            for line in self.lines:
                line.set_color(0, 0, 0)

            # 障碍物
            barrier_1 = rendering.make_polygon([(140, 140),
                                                (140, 260),
                                                (260, 260),
                                                (260, 140)], filled=True)
            barrier_2 = rendering.make_polygon([(260, 140),
                                                (380, 140),
                                                (380, 260),
                                                (260, 260)], filled=True)
            barrier_1.set_color(0, 0, 0)
            barrier_2.set_color(0, 0, 0)

            # 探索者
            explorer = rendering.FilledPolygon([(0, 0), (self.exploer_width, 0),
                                                (self.exploer_width, self.exploer_width), (0, self.exploer_width)])
            self.exploer_rans = rendering.Transform()
            explorer.add_attr(self.exploer_rans)
            explorer.set_color(1, 0, 0)

            # 宝藏
            treasure = rendering.make_circle(self.treasure_radius)
            circultrans = rendering.Transform(translation=(320, 320))
            treasure.add_attr(circultrans)
            treasure.set_color(0, 1, 0)

            for line in self.lines:
                self.viewer.add_geom(line)
            self.viewer.add_geom(barrier_1)
            self.viewer.add_geom(barrier_2)
            self.viewer.add_geom(treasure)
            self.viewer.add_geom(explorer)

        if self.state is None:
            return None

        x, y = self.pos[self.state]
        self.exploer_rans.set_translation(x, y)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
