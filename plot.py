import pickle
import numpy as np
import matplotlib.pyplot as plt

loss = []
with open("./loss_data/loss.pickle",'rb') as fr:
    while True:
        try:
            data = pickle.load(fr)
            loss.append(data)
        except EOFError:
            break
loss = np.array(loss)

plt.plot(loss)
# plt.ylim(0,5)
plt.show()