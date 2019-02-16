import sys
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


def get_data(feats, answers, major_note_index, samplerate, soundlen=15, split=1):
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

    # acceptable interval in seconds
    minspace = 0.1
    maxspace = 0.7
    idx = np.random.permutation(major_note_index.shape[0] - soundlen) + soundlen // 2  # 最初と最後の7フレームは消す
    X = []
    y = []

    for i in range(int(idx.shape[0]*split)):
        dist = major_note_index[idx[i] + 1] - major_note_index[idx[i]]  # notes間が0.1~0.7のもの
        
        if dist < maxspace * samplerate / 512 and dist > minspace * samplerate / 512:    
            for k in range(-1, dist+2):
                X.append(feats[:,:,major_note_index[idx[i]] - soundlen // 2 + k:major_note_index[idx[i]] + soundlen // 2 + k + 1])  # 15フレーム分
                y.append([answers[major_note_index[idx[i]] + k]])
    
    return X, y
    

def data_builder(songs, soundlen, split=1):
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
        X, y = get_data(song.feats, song.answer, song.major_note_index, song.samplerate, soundlen, split)
        Xs += X
        ys += y

    data = tuple_dataset.TupleDataset(np.array(Xs, dtype='float32'), np.array(ys, dtype='float32'))

    return data


"""
Training Process Flow:
    1. from songs, make teacher values(answer).
    2. milden answer; put smaller value to plus minus i frame.
    3. build chainer model; 256 mini-batch, and 300 epochs
    4. build chainer trainer.
    5. train chainer model.
"""

def train(songs, epochs, soundlen, saveplace, don_ka=0, split=1):
    """
    Args:
        songs: the list of song
        epochs: number of train 
        soundlen: width of one train data's image
        don-ka: don(1) or ka(2) or both(0), usually, firstly, train don, then, train ka.
    """


    """
    create answer values:
    On the papaer,
    To be used as an onset detector, we train a CNN on spectrogram
    excerpts centered on the frame to classify, giving binary labels
    to distinguish onsets from non-onsets.

    so we create onset/non-onset answer labels, 
    but to enhence model accuracy, we use answer values
    via small value to near answer labels
    """

    for song in songs:
        
        timing = song.timestamp[:, 0]
        sound = song.timestamp[:, 1]
        song.answer = np.zeros((song.feats.shape[2]))  # frameで切り分けた学習データの数分の答えを用意する

        # don/ka diteciton and make teacher values
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
    train = data_builder(songs, soundlen, split)
    test = data_builder(songs, soundlen, split)
    train_iter = iterators.SerialIterator(train, 128)
    test_iter = iterators.SerialIterator(test, 128, repeat=False, shuffle=False)
    optimizer = O.SGD(lr=0.02).setup(model)  # tried: 0.01, 0.02, 0.005
    # optimizer = O.Adam().setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)

    # build trainer
    trainer = training.Trainer(updater, (epochs, 'epoch'), out='./data/log')
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ParameterStatistics(model))
    trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time', 'main/loss']))
    chainer.config.train = True
    trainer.run()

    serializers.save_npz(saveplace, model) # npz形式でモデルをsave

    return model.predictor


if __name__ == '__main__':

    epochs = 70
    
    with open('./data/pickles/train_data.pickle', mode='rb') as f:
        songs = pickle.load(f)

    if sys.argv[1] == 'don':
        train(songs=songs, epochs=epochs, soundlen=15, saveplace="./models/don_model.npz", don_ka=1, split=0.2)

    if sys.argv[1] == 'ka':
        train(songs=songs, epochs=epochs, soundlen=15, saveplace="./models/ka_model.npz", don_ka=2, split=0.2)

    if sys.argv[1] == 'both':
        train(songs=songs, epochs=epochs, soundlen=15, saveplace="./models/both_model.npz", don_ka=0, split=0.2)

