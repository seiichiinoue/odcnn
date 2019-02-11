import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Chain
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

class convNet(Chain):
	"""
	copies the neural net used in a paper 
	"Improved musical onset detection with Convolutional Neural Networks".
	https://ieeexplore.ieee.org/document/6854953
	
	Args:

	
	Example:

	"""

	