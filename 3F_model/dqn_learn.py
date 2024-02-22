import numpy as np
import matplotlib.pyplot as plt
import random
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from replaybuffer import ReplayBuffer


# Q network
class DQN(Model):

    def __init__(self, action_n):
        super(DQN, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(action_n, activation='linear')


    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        q = self.q(x)
        return q


class DQNagent(object):

    def __init__(self, env, ratio=1):

        ## hyperparameters
        self.ratio = ratio
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.DQN_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.995
        self.EPSILON_MIN = 0.01

        self.env = env

        # get state dimension and action number
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = env.observation_space
        # self.action_n = env.action_space.n
        self.action_n = env.action_space

        ## create Q networks
        self.dqn = DQN(self.action_n)
        self.target_dqn = DQN(self.action_n)

        self.dqn.build(input_shape=(None, self.state_dim))
        self.target_dqn.build(input_shape=(None, self.state_dim))

        self.dqn.summary()

        # optimizer
        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []


    ## get action
    def choose_action(self, state):
        if np.random.random() <= self.EPSILON:
            # return self.env.action_space.sample()
            return random.randint(0,6)
        else:
            qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
            return np.argmax(qs.numpy())


    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        phi = self.dqn.get_weights()
        target_phi = self.target_dqn.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_dqn.set_weights(target_phi)


    ## single gradient update on a single batch data
    def dqn_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_n)
            q = self.dqn(states, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.dqn_opt.apply_gradients(zip(grads, self.dqn.trainable_variables))


    ## computing TD target: y_k = r_k + gamma* max Q(s_k+1, a)
    def td_target(self, rewards, target_qs, dones):
        max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * max_q[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.dqn.load_weights(path + 'cartpole_dqn.h5')


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset()
            self.save_epi_action = []
            self.save_epi_temp = []
            while not done:
                # visualize the environment
                #self.env.render()
                # pick an action
                action = self.choose_action(state)
                # observe reward, new_state
                next_state, reward, done, _ = self.env.step(action)

                train_reward = reward + time*0.01

                # add transition to replay buffer
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:  # start train after buffer has some amounts

                    # decaying EPSILON
                    if self.EPSILON > self.EPSILON_MIN:
                        self.EPSILON *= self.EPSILON_DECAY

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # predict target Q-values
                    target_qs = self.target_dqn(tf.convert_to_tensor(
                                                        next_states, dtype=tf.float32))

                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    # train critic using sampled batch
                    self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                   actions,
                                   tf.convert_to_tensor(y_i, dtype=tf.float32))


                    # update target network
                    self.update_target_network(self.TAU)
                    self.save_epi_action.append(action)
                    self.save_epi_temp.append(next_state[0])

                # update current state
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## save weights every episode
            try:
                self.dqn.save_weights('./'+ self.env.env_name + f'_{self.ratio:.1f}' + "/save_weights/cartpole_dqn.h5")
            except:
                os.mkdir('./' + self.env.env_name + f'_{self.ratio:.1f}')
                os.mkdir('./' + self.env.env_name + f'_{self.ratio:.1f}' + "/save_weights")

        np.savetxt('./'+ self.env.env_name + f'_{self.ratio:.1f}' +'/save_weights/cartpole_epi_reward.txt', self.save_epi_reward)

    def myEWMA(self, data, span):
        # 지수 이동 평균을 계산해서 저장할 리스트
        ewma=[0]*len(data)
        # 지수 이동 평균의 분자
        molecule=0
        # 지수 이동 평균의 분모
        denominator=0
        # 값에 곱해지는 가중치
        alpha = 2.0 / (1.0 + span)

        for i in range(len(data)):
            # 분자 계산 data+(1-alpha)앞의 데이터
            molecule = (data[i] + (1.0-alpha)*molecule)
            # 분모 계산 (1-alpha)의 i승
            denominator+=(1-alpha)**i
            # 지수 이동 평균 계산
            ewma[i] = molecule/denominator
        return ewma

    def plot_result(self, env_name):
        plt.plot(self.save_epi_reward, alpha=0.5)
        plt.plot(self.myEWMA(self.save_epi_reward,10), color='red')
        plt.savefig(env_name + f'_{self.ratio:.1f}' + '.jpg')
        # plt.show()
        plt.clf()

    def plot_simul(self):
        plt.figure(figsize=(15,10))
        plt.subplot(2,1,1)
        plt.plot(self.save_epi_temp, label='temp')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(self.save_epi_action, label='action')
        plt.legend()
        plt.show()