import gym
import math
import numpy as np

class generateData():
    def __init__(self,env):
        self.env = env
        #currently only writing for pendulum

    def do_n_steps(self,n,param_vec):
        data = []
        for param in param_vec:
            for i in range(n):
                data.append(self.step(param))
        return data


    def step(self,param):
        #sample param
        st = self.sample_state()
        a = self.sample_action()

        self.set_param(param)
        self.set_state(st)
        self.env.step(a)
        next_st = self.get_state()

        return (param,st,a,next_st)

    def set_param(self,param):
        #currently written for setting mass of pendulum
        self.env.unwrapped.m = param

    def set_state(self,st):
        self.env.unwrapped.state = st

    def get_state(self):
        return self.env.unwrapped.state

    def generate_param_vec(self,length):
        param_vec = []
        for i in range(length):
            param_vec.append(self.sample_param())

        return param_vec

    def sample_param(self):
        #currently sampling mass in range [0.5, 1.5]
        return np.random.rand()+0.5

    def sample_state(self):
        #currently sampling uniform
        th = self.angle_normalize(np.random.rand()*2*math.pi)
        thdot = (np.random.rand()-0.5)*2*self.env.unwrapped.max_speed
        return np.array([th, thdot])

    def sample_action(self):
        return env.action_space.sample()

    def angle_normalize(self,x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.reset()
    gen = generateData(env)
    param_vec = gen.generate_param_vec(5)
    print(gen.do_n_steps(5,param_vec))