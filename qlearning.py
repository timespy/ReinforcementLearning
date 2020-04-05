import gym
import numpy
import time
import random
import LineChart as dv
import numpy as np
import pandas as pd


class Q_learn:
    def __init__(self, states, actions, epoches, discount=0.9, learning_rate=0.1,
                 greedy=0.9, max_iteration=100, fresh_time=0.1):
        self.Q_table = pd.DataFrame(data=np.zeros((len(states), len(actions))), columns=actions, index=states)
        self.actions = actions
        self.states = states
        self.EPOCHES = epoches
        self.discount = discount
        self.lr = learning_rate
        self.greedy = greedy
        self.max_iteration = max_iteration
        self.fresh_time = fresh_time

    def exist_zero(self, state):
        for act in self.actions:
            if self.Q_table.loc[state, act] == 0:
                return True
        return False

    def choose_action(self, state):
        if state not in self.states:
            return self.actions[random.randint(0, len(self.actions)-1)]
        else:
            if random.uniform(0, 1) < self.greedy:
                if self.exist_zero(state):
                    return self.actions[random.randint(0, len(self.actions)-1)]
                else:
                    ret = None
                    max_q = 0.0
                    for act in self.actions:
                        if self.Q_table.loc[state, act] > max_q:
                            max_q = self.Q_table.loc[state, act]
                            ret = act
                    return ret
            else:
                return self.actions[random.randint(0, len(self.actions)-1)]

    def maxQ(self, state):
        max_q = 0.0
        for act in self.actions:
            if self.Q_table.loc[state, act] > max_q:
                max_q = self.Q_table.loc[state, act]
        return max_q

    def rl(self, env):
        x = []
        y = []
        for epoch in range(self.EPOCHES):
            observation = env.reset()
            count = 0
            ok = False
            for iteration in range(self.max_iteration):
                action = self.choose_action(observation)
                new_state, reward, done, info = env.step(action)
                self.Q_table.loc[observation, action] = (1 - self.lr) * self.Q_table.loc[observation, action]\
                                                        + self.lr * (reward + self.discount * self.maxQ(new_state))
                if done:
                    if reward == 1:
                        x.append(epoch)
                        y.append(count)
                        ok = True
                    break
                env.render()
                time.sleep(self.fresh_time)
                count += 1
                observation = new_state
            if ok:
                print(str(epoch) + ": win")
            else:
                print(str(epoch) + ": lose")

            # if epoch % 20 == 0:
            #     self.output_Q_table()
        env.close()
        return x, y

    def output_Q_table(self):
        print(self.Q_table)


if __name__ == '__main__':
    env = gym.make('MazeEnv-v0')
    test = Q_learn(env.getStates(), env.action_space, 150, fresh_time=0.1)
    x, y = test.rl(env)
    test.output_Q_table()
    pic = dv.LineChart()
    pic.add_line(x, y, 'g-s')
    pic.draw()