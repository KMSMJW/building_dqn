# DQN load and play
# coded by St.Watermelon

import numpy as np
import tensorflow as tf
from dqn_learn import DQNagent
from Environment import Environment
import matplotlib.pyplot as plt
import pandas as pd
import math
import dataframe_image as dfi

def action_count(action_list):
    result = 0
    for i in range(len(action_list)-1):
        if action_list[i+1] - action_list[i] > 0:
            result += (action_list[i+1] - action_list[i])
    return result

def reward_temp(state):
    if state[0] > 25:
        reward = 25 - state[0]
    elif state[0] < 24:
        reward = state[0] - 24
    else:
        reward = 0
    return reward

def compare_result():
    pd.options.display.float_format = '{: .2f}'.format
    df = pd.read_csv('test.csv')
    simul_list = ['dnn','formula','LGBM','LR']
    level1 = []
    for i in simul_list:
        for j in range(4):
            level1.append(i)
    level2 = []
    for j in range(4):
        level2.append('RL')
        level2.append('rule1')
        level2.append('rule2')
        level2.append('rule3')
    result = pd.DataFrame(columns=[level1,level2])
    for ratio in [0.1, 0.3, 0.5]:
        for env_name in simul_list:
            env1 = Environment(env_name, test=True, ratio = ratio)
            env2 = Environment(env_name, test=True, ratio = ratio)
            env3 = Environment(env_name, test=True, ratio = ratio)
            env4 = Environment(env_name, test=True, ratio = ratio)

            agent = DQNagent(env1,ratio)
            agent.load_weights('./' + env_name + f'_{ratio}' + '/save_weights/')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['Date'] = df['timestamp'].dt.date
            day_list = np.unique(df['Date'].astype('str'))

            reward_list1 = []
            reward_list2 = []
            reward_list3 = []
            reward_list4 = []
            action_list1 = []
            action_list2 = []
            action_list3 = []
            action_list4 = []
            action_num1 = []
            action_num2 = []
            action_num3 = []
            action_num4 = []

            for today in day_list:
                time = 0
                state1 = env1.reset(today)
                state2 = env2.reset(today)
                state3 = env3.reset(today)
                state4 = env4.reset(today)
                re1 = []
                re2 = []
                re3 = []
                re4 = []
                ac1 = []
                ac2 = []
                ac3 = []
                ac4 = []

                while True:
                    # env.render()
                    qs = agent.dqn(tf.convert_to_tensor([state1], dtype=tf.float32))
                    action1 = np.argmax(qs.numpy())
                    if state2[0] >= 24:
                        action2 = 5
                    else:
                        action2 = 0
                    if time%3 == 0:
                        action3 = 5
                    else:
                        action3 = 0
                    if state4[0] > 23:
                        deltaT = state2[0] - 23
                        num_ac = deltaT/0.29
                        action4 = np.min([math.ceil(num_ac), 5])
                    else:
                        action4 = 0
                    state1, reward1, done1, _ = env1.step(action1)
                    state2, reward2, done2, _ = env2.step(action2)
                    state3, reward3, done3, _ = env3.step(action3)
                    state4, reward4, done4, _ = env4.step(action4)
                    time += 1

                    re1.append(reward_temp(state1))
                    re2.append(reward_temp(state2))
                    re3.append(reward_temp(state3))
                    re4.append(reward_temp(state4))

                    ac1.append(action1)
                    ac2.append(action2)
                    ac3.append(action3)
                    ac4.append(action4)

                    if done1:
                        reward_list1.append(np.average(re1))
                        reward_list2.append(np.average(re2))
                        reward_list3.append(np.average(re3))
                        reward_list4.append(np.average(re4))
                        action_list1.append(action_count(ac1))
                        action_list2.append(action_count(ac2))
                        action_list3.append(action_count(ac3))
                        action_list4.append(action_count(ac4))
                        action_num1.append(np.sum(ac1))
                        action_num2.append(np.sum(ac2))
                        action_num3.append(np.sum(ac3))
                        action_num4.append(np.sum(ac4))
                        break
            result.loc['temp',env_name] = [np.average(reward_list1), np.average(reward_list2), np.average(reward_list3), np.average(reward_list4)]
            result.loc['OnOff',env_name] = [np.average(action_list1), np.average(action_list2), np.average(action_list3), np.average(action_list4)]
            result.loc['run_num',env_name] = [np.average(action_num1), np.average(action_num2), np.average(action_num3), np.average(action_num4)]
        dfi.export(result, f'5F_{ratio}.png', max_cols=-1, max_rows=-1)
 

def scoring():
    df = pd.read_csv('test.csv')
    env_name = 'LR'
    env1 = Environment(env_name, test=True)
    env2 = Environment(env_name, test=True)
    env3 = Environment(env_name, test=True)

    agent = DQNagent(env1)
    agent.load_weights('./' + env_name +'/save_weights/')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Date'] = df['timestamp'].dt.date
    day_list = np.unique(df['Date'].astype('str'))
    reward_list1 = []
    reward_list2 = []
    reward_list3 = []
    action_list1 = []
    action_list2 = []
    action_list3 = []
    for today in day_list:
        time = 0
        state1 = env1.reset(today)
        state2 = env2.reset(today)
        state3 = env3.reset(today)
        re1 = []
        re2 = []
        re3 = []
        ac1 = []
        ac2 = []
        ac3 = []

        while True:
            # env.render()
            qs = agent.dqn(tf.convert_to_tensor([state1], dtype=tf.float32))
            action1 = np.argmax(qs.numpy())
            if state2[0] >= 24:
                action2 = 5
            else:
                action2 = 0
            if time%3 == 0:
                action3 = 5
            else:
                action3 = 0
            state1, reward1, done1, _ = env1.step(action1)
            state2, reward2, done2, _ = env2.step(action2)
            state3, reward3, done3, _ = env3.step(action3)
            time += 1

            re1.append(reward1)
            re2.append(reward2)
            re3.append(reward3)

            ac1.append(action1)
            ac2.append(action2)
            ac3.append(action3)

            if done1:
                reward_list1.append(np.average(re1))
                reward_list2.append(np.average(re2))
                reward_list3.append(np.average(re3))
                action_list1.append(action_count(ac1))
                action_list2.append(action_count(ac2))
                action_list3.append(action_count(ac3))
                break
    # print("reward1:", reward_list1)
    print(f"reward1 avg:{np.average(reward_list1)}")
    print(f"action1 count avg: {np.average(action_list1)}")
    # print("reward2:", reward_list2)
    print(f"reward2 avg:{np.average(reward_list2)}")
    print(f"action2 count avg: {np.average(action_list2)}")
    # print("reward3:", reward_list3)
    print(f"reward3 avg:{np.average(reward_list3)}")
    print(f"action3 count avg: {np.average(action_list3)}")

def main():

    env_name = 'formula'
    # env = gym.make(env_name)
    env1 = Environment(env_name)
    env2 = Environment(env_name)
    env3 = Environment(env_name)

    # print(env.observation_space.shape[0])  # 4
    # get action dimension
    # print(env.action_space, env.observation_space)

    agent = DQNagent(env1)

    agent.load_weights('./' + env_name +'/save_weights/')

    time = 0
    state1 = env1.reset()
    today = env1.today
    state2 = env2.reset(today)
    state3 = env3.reset(today)
    action_list1 = []
    action_list2 = []
    action_list3 = []
    temp1 = []
    temp2 = []
    temp3 = []

    while True:
        # env.render()
        qs = agent.dqn(tf.convert_to_tensor([state1], dtype=tf.float32))
        action1 = np.argmax(qs.numpy())
        if state2[0] >= 25:
            action2 = 5
        else:
            action2 = 0
        if time%3 == 0:
            action3 = 5
        else:
            action3 = 0
        state1, reward1, done1, _ = env1.step(action1)
        state2, reward2, done2, _ = env2.step(action2)
        state3, reward3, done3, _ = env3.step(action3)
        time += 1

        print('Time: ', time, 'Reward: ', reward1)
        action_list1.append(action1)
        action_list2.append(action2)
        action_list3.append(action3)
        temp1.append(state1[0])
        temp2.append(state2[0])
        temp3.append(state3[0])

        if done1:
            break

    # env.close()
    print(today)
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(action_list1, label='action1')
    plt.plot(action_list2, label='action2')
    plt.plot(action_list3, label='action3')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(temp1, label='temp1')
    plt.plot(temp2, label='temp2')
    plt.plot(temp3, label='temp3')
    plt.axhline(22,0,1,color='red',linestyle='--')
    plt.axhline(23,0,1,color='red',linestyle='--')
    plt.legend()
    plt.show()

if __name__=="__main__":
    # main()
    # scoring()
    compare_result()