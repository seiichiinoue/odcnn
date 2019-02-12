import numpy as np
import pandas as pd
import chainer
from tqdm import tqdm
from glob import glob
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer.datasets import tuple_dataset
from chainer import datasets, iterators, serializers
from chainer import Link, Chain, ChainList
from chainer.functions.loss.mean_squared_error import mean_squared_error
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer.training import extensions
# my modules
from models import convNet
from music_processor import *

def get_data(feats, answers, major_note_index, samplerate, soundlen=15):
    """
    Args:
        feats: song.feats; Audio module
        answers: song.answers; Audio module
        major_note_index: answer labels; corresponding to feats
        samplerate: song.samplerate; Audio module
        soundlen: =15. 学習モデルに渡す画像データの横方向の長さ．ここでは(80 * 15)サイズのデータを使用している
        split: =1. 
    Variables:
        minspace: minimum space between major note indexs
        maxspace: maximum space between major note indexs
        idx: index of major_note_index or feats
        dist: distance of two notes
    """

    minspace = 0.1
    maxspace = 0.7
    idx = np.random.permutation(major_note_index.shape[0] - soundlen) + soundlen // 2
    X = []
    y = []

    for i in range(int(idx.shape[0])):
        dist = major_note_index[idx[i] + 1] - major_note_index[idx[i]]
        
        if dist < maxspace * samplerate / 512 and dist > minspace * samplerate / 512:    
            for k in range(-1, dist+2):
                X.append(feats[:,:,major_note_index[idx[i]] - soundlen // 2 + k:major_note_index[idx[i]] + soundlen // 2 + k + 1])
                y.append([answers[major_note_index[idx[i]] + k]])
    
    return X, y
    

def data_builder(songs, soundlen):
    """helping function to build Chainer formated dataset.
    Args:
        songs: the list of song
        soundlen: width of one train data's image 
    Variables:
        data: the list of tuple; (X, y) for training with Chainer
    """

    Xs = []
    ys = []
    for song in songs:
        X, y = get_data(song.feats, song.answer, song.major_note_index, song.samplerate, soundlen)
        Xs += X
        ys += y

    data = tuple_dataset.TupleDataset(np.array(Xs, dtype='float32'), np.array(ys, dtype='float32'))

    return data


"""
Training Process Flow:
    1. from songs, make teacher labels(answer).
    2. milden answer.
    3. build chainer model.
    4. build chainer trainer.
    5. train model.
"""

def train(songs, epochs, soundlen, don_ka=0):
    """
    Args:
        songs: the list of song
        epochs: number of train 
        soundlen: width of one train data's image
        don-ka: don(1) or ka(2) or both(0)
    """

    for song in songs:
        
        print(song.timestamp)
        timing = song.timestamp[:, 0]
        sound = song.timestamp[:, 1]
        song.answer = np.zeros((song.feats.shape[2]))  # frameで切り分けた学習データの数分の答えを用意する

        # don/ka diteciton and make teacher labels
        if don_ka == 0:
            song.major_note_index = np.rint(timing[np.where(sound != 0)] * song.samplerate / 512).astype(np.int32)
        else:
            song.major_note_index = np.rint(timing[np.where(sound == don_ka)] * song.samplerate / 512).astype(np.int32)
            song.minor_note_index = np.rint(timing[np.where(sound == 3 - don_ka)] * song.samplerate / 512).astype(np.int32)
        song.major_note_index = np.delete(song.major_note_index, np.where(song.major_note_index >= song.feats.shape[2]))
        song.minor_note_index = np.delete(song.minor_note_index, np.where(song.minor_note_index >= song.feats.shape[2]))

        song.answer[song.major_note_index] = 1
        song.answer[song.minor_note_index] = 0.26
        song.answer = milden(song.answer)


    # build model
    model = L.Classifier(convNet(), lossfun=mean_squared_error)
    model.compute_accuracy = False  # not need accuracy cz teacher data is not label; using sigmoid func
    train = data_builder(songs, soundlen)
    test = data_builder(songs, soundlen)
    train_iter = iterators.SerialIterator(train, 256)
    test_iter = iterators.SerialIterator(test, 256, repeat=False, shuffle=False)
    optimizer = O.Adam().setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)

    # build trainer
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    chainer.config.train = True
    trainer.run()

    return model.predicrtor


if __name__ == '__main__':
    with open('./data/pickles/testdata.pickle', mode='rb') as f:
        songs = [pickle.load(f)]


    # with open('./data/pickles/traindata.pickle', mode='rb') as f:
    #     songs = pickle.load(f)

        train(songs=songs, epochs=30, soundlen=15)
