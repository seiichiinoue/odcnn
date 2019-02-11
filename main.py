import argparse
import numpy as np
import pandas as pd
import chainer
from tqdm import tqdm
from glob import glob
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer.training import extensions
# my modules
import model
from music_processor import *

def get_data(feats, answers, major_note_index,
				samplerate, soundlen=15, split=1):
	"""
	Args:
		feats: song.feats; Audio module
		answers: song.answers; Audio module
		major_note_index: 
		samplerate: song.samplerate; Audio module
		soundlen: =15.
		split: =1. 
	"""
	minspace = 0.1
	maxspace = 0.7
	idx = np.random.permutation(major_note_index.shape[0] - soundlen) + soundlen // 2
	data = []

	for i in range(int(idx.shape[0] * split)):
		
		dist = major_note_index[idx[i] + 1] - major_note_index[idx[i]]
		
		if dist < maxspace * samplerate / 512 and dist > minspace * samplerate / 512:
			
			for k in range(-1, dist+2):
				
				X = feats[:,:,major_note_index[idx[i]] - soundlen // 2 + k:
								major_note_index[idx[i]] + soundlen // 2 + k + 1]
				y = answers[major_note_index[idx[i]] + k]
				
				data.append((np.array(X), y))
	return data		
	

def data_builder(songs):
	train = []
	for song in songs:
		train.append(get_data(song.feats, song.answer, song.major_note_index, 
				song.samplerate, soundlen, split=0.2))

	return train



def train(songs, epochs, soundlen, don_ka=0):
	"""
	Args:
	"""

	for song in songs:
		
		timing = song.timestamp[:, 0]
		sound = song.timestamp[:, 1]
		song.answer = np.zeros((song.feats.shape[2]))

		if don_ka == 0:
			song.major_note_index = np.rint(timing[np.where(sound != 0)]
											* song.samplerate / 512).astype(np.int32)
		else:
			song.major_note_index = np.rint(timing[np.where(sound == don_ka)]
											* song.samplerate / 512).astype(np.int32)
			song.minor_note_index = np.rint(timing[np.where(sound == 3 - don_ka)]
											* song.samplerate / 512).astype(np.int32)
		song.major_note_index = np.delete(song.major_note_index, 
											np.where(song.major_note_index >= song.feats.shape[2]))
		song.minor_note_index = np.delete(song.minor_note_index,
											np.where(song.minor_note_index >= song.feats.shape[2]))

		song.answer[song.major_note_index] = 1
		song.answer[song.minor_note_index] = 0.26
		# song.answer = milden(song.answer)


	# build model
	model = L.Classifier(model.convNet())

	train = data_builder(songs)
	test = data_builder(songs)
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
	with open('./data/pickles/traindata.pickle', mode='rb') as f:
		songs = pickle.load(f)

	train(songs=songs, epochs=30, soundlen=15)

