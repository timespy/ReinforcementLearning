import gym
import time
import random
import LineChart as dv
import numpy as np
import pandas as pd


class Q_learn:
    def __init__(self, actions, epoches, discount=0.9, learning_rate=0.1,
                 greedy=0.9, max_iteration=100, fresh_time=0.1):
        self.Q_table = pd.DataFrame(columns=actions)
        self.actions = actions
        self.EPOCHES = epoches
        self.discount = discount
        self.lr = learning_rate
        self.greedy = greedy
        self.max_iteration = max_iteration
        self.fresh_time = fresh_time

    def exist_state(self, state):
        if state not in self.Q_table.index:
            self.Q_table = self.Q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.Q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, state):
        self.exist_state(state)
        if random.uniform(0, 1) < self.greedy:
            qs = self.Q_table.loc[state, :]
            action_ret = np.random.choice(qs[qs == np.max(qs)].index)
            return action_ret
        else:
            return self.actions[random.randint(0, len(self.actions)-1)]

    def rl(self, env):
        x = []
        y = []
        for epoch in range(self.EPOCHES):
            observation = env.reset()
            count = 0
            ok = False
            for iteration in range(self.max_iteration):
                env.render()

                action = self.choose_action(observation)
                new_state, reward, done, info = env.step(action)
                self.exist_state(new_state)
                self.Q_table.loc[observation, action] = (1 - self.lr) * self.Q_table.loc[observation, action]\
                                                        + self.lr * \
                                                        (reward + self.discount * self.Q_table.loc[new_state, :].max())
                if done:
                    if reward == 1:
                        x.append(epoch)
                        y.append(count)
                        ok = True
                    break

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
    test = Q_learn(env.action_space, 100, fresh_time=0.1)
    x, y = test.rl(env)
    test.output_Q_table()
    pic = dv.LineChart()
    pic.add_line(x, y, 'g-s')
    pic.draw()