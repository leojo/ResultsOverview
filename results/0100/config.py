import math
import os
import random

import numpy as np
import waveUtils


class config(object):
	# Bsub arguments
	bsub_mainfile = "myMain.py"
	bsub_processors = 4
	bsub_timeout = "4:00"
	bsub_memory = 8000

	# Epoch and batch config
	batch_size = 128
	latent_dim = 10
	epochs = 100
	epoch_updates = 100
	random_noise_stdev = 0

	# Network structure
	input_s = 16000
	n_ae = 2
	n_conv_layers = 3
	n_deconv_layers = 3
	first_size = input_s // (2 ** n_deconv_layers)
	final_decoder_filter_size = 3

	# Memmap parameters
	load_model = False
	model_path = os.path.join("LOCAL", "model", "model")  # only used if load_model=True
	original_sample_rate = 48000
	original_duration = 2

	# Miscellaneous constants
	min_freq = 100
	max_freq = 800
	num_freqs = 50
	sample_rate = 8000
	_val_data = None
	_val_originals = None
	time_loss_mult = 0.0005
	reconstruction_mult = 1
	learning_rate = 1e-3
	kl_loss_mult = 1e-3
	kl_extra_mult = 2
	kl_extra_exponent = 2 
	use_square = False
	data_sources = ["cello", "cello"]  # Must be exactly two
	data = None
	training_indices = [None, None]
	validation_batch = None

	# Functions
	def load_data(self):
		if self.data is None:
			self.data = [np.memmap(os.path.join("data", "{}.memmap".format(x)), dtype=np.float64, mode="r")
				             .reshape(-1, self.original_sample_rate * self.original_duration, 1) for x in self.data_sources]

	def get_data(self, i, j):
		self.load_data()
		return waveUtils.reduceQualityOfSingleWave(self.data[i][j], self.sample_rate, self.original_duration)[:self.input_s]

	def make_validation_batch(self):
		self.load_data()
		batch = []
		if self.data_sources[0] == self.data_sources[1]:
			num_val = int(math.ceil((math.sqrt(8*self.batch_size)+1)/2))
			p = np.random.permutation(len(self.data[0]))
			validation_indices = p[:num_val]
			self.training_indices = [p[num_val:], p[num_val:]]
			for i in range(len(validation_indices)):
				ii = validation_indices[i]
				for j in range(i+1, len(validation_indices)):
					jj = validation_indices[j]
					wave1 = self.get_data(0, ii)
					wave2 = self.get_data(1, jj)
					batch.append([wave1, wave2])
		else:
			num_val = int(math.ceil(math.sqrt(self.batch_size)))
			p0 = np.random.permutation(len(self.data[0]))
			p1 = np.random.permutation(len(self.data[1]))
			validation_indices = [p0[:num_val], p1[:num_val]]
			self.training_indices = [p0[num_val:], p1[num_val:]]
			for i in validation_indices[0]:
				for j in validation_indices[1]:
					wave1 = self.get_data(0, i)
					wave2 = self.get_data(1, j)
					batch.append([wave1, wave2])

		self.validation_batch = np.asarray(batch[:self.batch_size])


	def get_training_batch(self):
		samples = []
		originals = []
		for _ in range(self.batch_size):
			i = random.choice(self.training_indices[0])
			j = random.choice(self.training_indices[1])
			wave1 = self.get_data(0, i)
			wave2 = self.get_data(1, j)
			samples.append((wave1+wave2)/2.0)
			originals.append([wave1, wave2])

		samples = np.asarray(samples)
		originals = np.asarray(originals)
		return samples, originals

	def get_validation_batch(self):
		if self.validation_batch is None:
			self.make_validation_batch()
		samples = []
		for pair in self.validation_batch:
			samples.append((pair[0]+pair[1])/2.0)

		samples = np.asarray(samples)
		return samples, self.validation_batch

	def normalize_batch(self, batch):
		x = batch.astype(np.float32)
		return x / np.max(np.abs(x))

	# return x / np.linalg.norm(x)

	def deconv_filter_size(self, i):
		return (2 * (i + 1)) + 1

	def deconv_channel_num(self, i):
		return 2 ** (config.n_deconv_layers + 3 - i)

	def conv_filter_size(self, i):
		return (2 * (config.n_conv_layers - i)) + 1

	def conv_channel_num(self, i):
		return 2 ** (i + 4)
