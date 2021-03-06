import math
import os

import numpy as np
import waveUtils


class config(object):
	# Bsub arguments
	bsub_mainfile = "main.py"
	bsub_processors = 4
	bsub_timeout = "4:00"
	bsub_memory = 4000

	# Epoch and batch config
	batch_size = 128
	latent_dim = 100
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
	original_sample_rate = 8000
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
	kl_extra_mult = 200
	kl_extra_exponent = 2 
	use_square = False
	data_sources = "clarinet-clarinet-8000-2"
	data = None
	training_indices = None
	validation_indices = None

	# Functions
	def load_data(self):
		if self.data is None:
			self.data = np.memmap(os.path.join("data", "{}.memmap".format(self.data_sources)), dtype=np.float64, mode="r")\
				.reshape(-1, 3, self.original_sample_rate * self.original_duration, 1)

	def make_validation_batch(self):
		self.load_data()
		p = np.random.permutation(len(self.data))
		self.validation_indices = p[:self.batch_size]
		self.training_indices = p[self.batch_size:]

	def get_training_batch(self):
		if self.training_indices is None:
			self.make_validation_batch()
		samples = []
		originals = []
		for _ in range(self.batch_size):
			i = np.random.choice(self.training_indices)
			originals.append([
				self.data[i, 0, :, :],
				self.data[i, 1, :, :],
			])
			samples.append(self.data[i, 2, :, :])

		samples = np.asarray(samples)
		originals = np.asarray(originals)
		return samples, originals

	def get_validation_batch(self):
		if self.validation_indices is None:
			self.make_validation_batch()
		samples = []
		originals = []
		for i in self.validation_indices:
			originals.append([
				self.data[i, 0, :, :],
				self.data[i, 1, :, :],
			])
			samples.append(self.data[i, 2, :, :])

		samples = np.asarray(samples)
		originals = np.asarray(originals)
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
