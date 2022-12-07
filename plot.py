import pickle
import numpy as np
import matplotlib.pyplot as plt


try:
    loss = []
    with open("./loss_data/loss.pickle",'rb') as fr:
        while True:
            try:
                data = pickle.load(fr)
                loss.append(data)
            except EOFError:
                break
    loss = np.array(loss)
    plt.subplot(2,2,1)
    plt.plot(loss)
    # plt.ylim(0,5)
    # plt.show()
except FileNotFoundError:
    pass

try:
    reward = []
    with open("./reward_data/reward.pickle",'rb') as fr:
        while True:
            try:
                data = pickle.load(fr)
                reward.append(data)
            except EOFError:
                break
    reward = np.array(reward)
    plt.subplot(2,2,2)
    plt.plot(reward)
except FileNotFoundError:
    pass

try:
    critic = []
    with open("./reward_data/critic_loss.pickle",'rb') as fr:
        while True:
            try:
                data = pickle.load(fr)
                critic.append(data)
            except EOFError:
                break
    critic = np.array(critic)
    plt.subplot(2,2,3)
    plt.plot(critic)
except FileNotFoundError:
    pass

try:
    actor = []
    with open("./reward_data/actor_loss.pickle",'rb') as fr:
        while True:
            try:
                data = pickle.load(fr)
                actor.append(data)
            except EOFError:
                break
    actor = np.array(actor)
    plt.subplot(2,2,4)
    plt.plot(actor)
    # plt.ylim(0,5)
except FileNotFoundError:
    pass
plt.show()