import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

"""
On the paper,
Starting from a stack of three spectrogram excerpts,
convolution and max-pooling in turns compute a set of 20 feature maps 
classified with a fully-connected network.
"""

class convNet(Chain):
	"""
	copies the neural net used in a paper 
	"Improved musical onset detection with Convolutional Neural Networks".
	https://ieeexplore.ieee.org/document/6854953
	
	Args:

	
	Example:

	"""

	def __init__(self):
		super(convNet, self).__init__()
		with self.init_scope():
			# convolution
			self.conv1 = L.Convolution2D(3, 10, ksize=(3,7))
			self.conv2 = L.Convolution2D(10, 20, 3)
			# full connection
			self.fc1 = L.Linear(1120, 256)
			self.fc2 = L.Linear(256, 120)
			self.fc3 = L.Linear(120, 1)


	def __call__(self, x, minibatch=1):
		x = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=(3,1))
		x = F.max_pooling_2d(F.relu(self.conv2(x)), ksize=(3,1))
		x = F.dropout(np.reshape(x, [minibatch, -1]))
		x = F.dropout(F.relu(self.fc1(x)))
		x = F.dropout(F.relu(self.fc2(x)))
		
		return F.sigmoid(self.fc3(x))




	