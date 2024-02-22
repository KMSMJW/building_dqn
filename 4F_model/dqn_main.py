# DQN main
# coded by St.Watermelon

from dqn_learn import DQNagent
from Environment import Environment
import time

def main():
    start = time.time()
    max_episode_num = 3000
    env_name = 'LR'
    # env = gym.make(env_name)
    env = Environment(env_name)
    agent = DQNagent(env)

    agent.train(max_episode_num)
    end = time.time()
    print(f'소요시간 : {end-start:.5f}sec')
    print('env name: ' + env_name)
    agent.plot_result(env_name)

def total_train():
    for ratio in [0.1, 0.3, 0.5]:
        for env_name in ['dnn', 'formula', 'LGBM', 'LR']:
            max_episode_num = 3000
            env = Environment(env_name=env_name, ratio=ratio)
            agent = DQNagent(env=env, ratio=ratio)

            agent.train(max_episode_num)
            print('env_name:' + env_name)
            print('---------------------- finish ------------------------')
            agent.plot_result(env_name)

if __name__=="__main__":
    # main()
    total_train()