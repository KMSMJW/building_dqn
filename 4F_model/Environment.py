import pandas as pd
import numpy as np
from state_observation import State
import warnings
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment',  None)

class Environment():
    def __init__(self,env_name,test=False, ratio = 1):
        # self.state = State()
        self.test = test
        self.ratio = ratio
        self.env_name = env_name
        if env_name == 'formula':
            self.observation_space = 7
        else:
            self.observation_space = 6
        self.action_space = 6
        self.timestep = 30
        if env_name == 'dnn':
            self.model = tf.keras.models.load_model('./dnn/model')
            self.scaler_x = joblib.load('./dnn/scaler_x')
            self.scaler_y = joblib.load('./dnn/scaler_y')
        elif env_name == 'LR':
            self.model = pickle.load(open('./LR/model','rb'))
            self.scaler_x = joblib.load('./LR/scaler_x')
            self.scaler_y = joblib.load('./LR/scaler_y')
        elif env_name == 'LGBM':
            self.model = pickle.load(open('./LGBM/model','rb'))
            self.scaler_x = joblib.load('./LGBM/scaler_x')
            self.scaler_y = joblib.load('./LGBM/scaler_y')
        elif env_name == 'formula':
            self.model = pickle.load(open('./formula/model','rb'))
            self.scaler_x = joblib.load('./formula/scaler_x')
            self.scaler_y = joblib.load('./formula/scaler_y')

    def make_csv(self, today=None, test=False):
        # df = pd.read_csv('total_merge.csv')
        if self.test:
            df = pd.read_csv('test.csv')
        else:
            df = pd.read_csv('train.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['timestamp'].dt.date
        day_list = np.unique(df['Date']).astype('str')
        df['hour'] = df['timestamp'].dt.hour
        df['cnt'] = (df['timestamp'].dt.hour*60 + df['timestamp'].dt.minute)//self.timestep
        df = df.groupby(['Date','cnt']).mean()
        df = df.reset_index()
        if today == None:
            self.today = np.random.choice(day_list,1)[0]
        else:
            self.today = today
        df['Date'] = df['Date'].astype('str')
        df1 = df[df['Date'] == self.today]
        df1 = df1[(df1['hour']<=22) & (df1['hour']>=8)].reset_index(drop=True)
        df1 = df1[['4F_temp','TA','CA','instant_headcount','cumulative_headcount','cnt']]
        if self.env_name == 'formula':
            df1['up'] = 0
            df1['down'] = 0
            for i in range(df1.shape[0]):
                if df1['4F_temp'].loc[i] >= df1['TA'].loc[i]:
                    df1['up'].loc[i] = df1['4F_temp'].loc[i] - df1['TA'].loc[i]
                else:
                    df1['down'].loc[i] = df1['TA'].loc[i] - df1['4F_temp'].loc[i]
            df1 = df1[['4F_temp','up','down','CA','instant_headcount','cumulative_headcount','cnt']]
        return df1

    def reset(self,today=None):
        self.df1 = self.make_csv(today)
        self.state = self.df1.loc[0]
        return self.df1.loc[0]
    
    def step(self,action):
        cnt = self.state['cnt']
        reward_action = action
        action = action*6000/9.7
        input = np.append(np.array(self.state), action).reshape(1,-1)
        input = self.scaler_x.transform(input)
        if self.env_name == 'dnn':
            next_ra = self.model(input)
        elif self.env_name == 'LR' or self.env_name == 'formula':
            next_ra = self.model.predict(input)
        elif self.env_name == 'LGBM':
            next_ra = self.model.predict(input).reshape(-1,1)
        next_ra = self.scaler_y.inverse_transform(next_ra)
        next_state = self.df1.loc[cnt-15]
        next_state[0] = next_ra[0,0]
        if next_ra[0,0] > 24:
            reward = 24 - next_ra[0,0]
        elif next_ra[0,0] < 23:
            reward = next_ra[0,0] - 23
        else:
            reward = 0
        reward = reward - reward_action*self.ratio
        if cnt == 44:
            done = True
        else:
            done = False
        drop = 0
        self.state = next_state
        return next_state, reward, done, drop