import math
import os
import random

import numpy as np
import waveUtils


class config(object):
	# Bsub arguments
	bsub_mainfile = "main.py"
	bsub_processors = 4
	bsub_timeout = "4:00"
	bsub_memory = 8000

	# Epoch and batch config
	batch_size = 128
	latent_dim = 100
	epochs = 200
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
	model_path = os.path.join("models", "0103", "model")  # only used if load_model=True
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
	kl_loss_mult = 1e-8
	kl_extra_mult = 200
	kl_extra_exponent = 2 
	use_square = False
	data_sources = ["cello_bad", "clarinet"]  # Must be exactly two
	data = None
	training_indices = None
	validation_indices = None

	# Functions
	def load_and_prepare_audio(self, source):
		duration = self.input_s / float(self.sample_rate)
		data_dir = os.path.join("wav_files", source)
		waves, original_sample_rate = waveUtils.loadAudioFiles(data_dir)
		cut_data = waveUtils.extractHighestMeanIntensities(waves, sample_rate=original_sample_rate, duration=duration)
		del waves
		data = waveUtils.reduceQuality(cut_data, self.sample_rate, duration)
		del cut_data
		return data

	def load_data(self):
		if self.data is None:
			self.data = [self.load_and_prepare_audio(source) for source in self.data_sources]

	def make_validation_batch(self):
		self.load_data()
		if len(self.data_sources) == 1:
			num_val = int(math.ceil((math.sqrt(8 * self.batch_size) + 1) / 2))
			p = np.random.permutation(len(self.data[0]))
			self.validation_indices = [p[:num_val], p[:num_val]]
			self.training_indices = [p[num_val:], p[num_val:]]
		else:
			num_val = int(math.ceil(math.sqrt(self.batch_size)))
			p0 = np.random.permutation(len(self.data[0]))
			p1 = np.random.permutation(len(self.data[1]))
			self.validation_indices = [p0[:num_val], p1[:num_val]]
			self.training_indices = [p0[num_val:], p1[num_val:]]

	def get_training_batch(self):
		if self.training_indices is None:
			self.make_validation_batch()
		samples = []
		originals = []
		for _ in range(self.batch_size):
			i = np.random.choice(self.training_indices[0])
			j = np.random.choice(self.training_indices[1])
			wave1 = self.data[0][i]
			wave2 = self.data[1][j]
			samples.append((wave1 + wave2) / 2.0)
			originals.append([wave1, wave2])

		samples = np.asarray(samples)
		originals = np.asarray(originals)
		return samples, originals

	def get_validation_batch(self):
		if self.validation_indices is None:
			self.make_validation_batch()
		samples = []
		originals = []
		p1 = np.random.permutation(self.validation_indices[0])
		p2 = np.random.permutation(self.validation_indices[1])
		for i in p1:
			for j in p2:
				wave1 = self.data[0][i]
				wave2 = self.data[1][j]
				samples.append((wave1 + wave2) / 2.0)
				originals.append([wave1, wave2])
		samples = np.asarray(samples)[:self.batch_size]
		originals = np.asarray(originals)[:self.batch_size]
		return samples, originals

	def normalize_batch(self, batch):
		x = batch.astype(np.float32)
		return x / np.max(np.abs(x))

	# return x / np.linalg.norm(x)

	def deconv_filter_size(self, i):
		return (2 * (i + 1)) + 3

	def deconv_channel_num(self, i):
		return 2 ** (config.n_deconv_layers + 3 - i)

	def conv_filter_size(self, i):
		return (2 * (config.n_conv_layers - i)) + 3

	def conv_channel_num(self, i):
		return 2 ** (i + 4)
