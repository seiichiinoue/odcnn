import chainer
import pickle
import matplotlib.pyplot as plt
from music_processor import *

if __name__ == '__main__':
    with open('./data/pickles/don_inference.pickle', mode='rb') as f:
        data = pickle.load(f) 

    plt.plot(range(len(data)), data)
    plt.show()
