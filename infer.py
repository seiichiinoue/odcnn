import sys
import pickle
import numpy as np
import pandas as pd
import chainer
from tqdm import tqdm
from glob import glob
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer.datasets import tuple_dataset
from chainer import datasets, iterators, serializers
from chainer import Link, Chain, ChainList
from chainer.functions.loss.mean_squared_error import mean_squared_error
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer.training import extensions
import matplotlib.pyplot as plt
# my modules
from models import convNet
from music_processor import *


def data_builder(feats, soundlen=15):
    x = []
    for i in range(feats.shape[2] - soundlen):
        x.append(feats[:,:,i:i+soundlen])

    return np.array(x, dtype='float32')


def infer(feats, saved_model, soundlen=15):
    
    model = L.Classifier(convNet(), lossfun=mean_squared_error)
    serializers.load_npz(saved_model, model)

    x = data_builder(feats, soundlen=15)

    return model.predictor(x)


def show(filename):

    with open(filename, mode='rb') as f:
        data = pickle.load(f) 

    plt.plot(range(len(data)), data)
    plt.show()


if __name__ == '__main__':
    
    with open('./data/pickles/testdata.pickle', mode='rb') as f:
        songs = pickle.load(f)
        
    if sys.argv[1] == 'don':
        result = infer(songs.feats, saved_model="./models/don_model.npz", soundlen=15)
        result = np.reshape(result.data, (-1))
        with open('./data/pickles/don_inference.pickle', mode='wb') as f:
            pickle.dump(result, f)
        show('./data/pickles/don_inference.pickle')

    if sys.argv[1] == 'ka':
        result = infer(songs.feats, saved_model="./models/ka_model.npz", soundlen=15)
        result = np.reshape(result.data, (-1))
        with open('./data/pickles/ka_inference.pickle', mode='wb') as f:
            pickle.dump(result, f)
        show('./data/pickles/ka_inference.pickle')

    if sys.argv[1] == 'both':
        result = infer(songs.feats, saved_model="./models/both_model.npz", soundlen=15)
        result = np.reshape(result.data, (-1))
        with open('./data/pickles/both_inference.pickle', mode='wb') as f:
            pickle.dump(result, f)

