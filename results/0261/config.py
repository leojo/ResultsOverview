import os
import numpy as np
import waveUtils


class config(object):

	def __init__(self):
		self.prepare_data()

	# Bsub arguments
	bsub_mainfile = "main.py"
	bsub_processors = 4
	bsub_timeout = "24:00"
	bsub_memory = 32000

	# Epoch and batch config
	batch_size = 32
	latent_dim = 1600
	epochs = 600
	epoch_updates = 100

	# Network structure
	input_s = 16384
	n_ae = 2

	# Model
	load_model = False
	model_path = os.path.join("models", "0230", "model")  # only used if load_model=True

	# Miscellaneous constants
	sample_rate = 8000
	learning_rate = 1e-4
	min_learning_rate = 1e-8
	learning_rate_reduce_threshold = 1
	learning_rate_reduction_count = 600
	kl_loss_mult = 1e-4
	kl_extra_mult = 2
	kl_extra_exponent = 2
	keep_prob = 1
	random_noise_stddev = 0
	useLogProba = False
	useMixtureProba = True
	deterministic = False
	reorder_outputs = False
	var_func = "relu"
	data_sources = ["speech","all_music"]
	validation_sources = ["speech","all_music"]
	data = None
	validation_data = None

	# Functions
	def prepare_data(self):
		self.load_data()

	def load_and_prepare_audio(self, source, i):
		duration = self.input_s / float(self.sample_rate)
		data_dir = os.path.join("wav_files", source)
		waves, original_sample_rate = waveUtils.loadAudioFiles(data_dir)
		if i == 0:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=12, sample_rate=original_sample_rate, duration=duration)
		elif i == 1:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=3, sample_rate=original_sample_rate, duration=duration, max_intensity=0.5)
		elif i == 2:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=6, sample_rate=original_sample_rate, duration=duration)
		elif i == 3:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=1, sample_rate=original_sample_rate, duration=duration, max_intensity=0.5)
		else:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=1, sample_rate=original_sample_rate, duration=duration)
		del waves
		data = waveUtils.reduceQuality(cut_data, self.sample_rate, duration)
		del cut_data
		return data

	def load_data(self):
		if self.data is None:
			self.data = [self.load_and_prepare_audio(source, i) for i, source in enumerate(self.data_sources)]
		if self.validation_data is None:
			self.validation_data = [self.load_and_prepare_audio(source, i+2) for i, source in enumerate(self.validation_sources)]

	def get_training_batch(self, add_noise=True):
		samples = []
		originals = []
		num_sources = len(self.data_sources)
		sample_shape = self.data[0][0].shape
		for _ in range(self.batch_size):
			waves = []
			sample = np.zeros(sample_shape)
			for s in range(num_sources):
				i = np.random.randint(len(self.data[s]))
				wave = self.data[s][i]
				waves.append(wave)
				sample += wave
			sample = sample/num_sources
			samples.append(sample)
			originals.append(waves)

		samples = np.asarray(samples)
		if add_noise and self.random_noise_stddev > 0:
			samples = samples + np.random.normal(0, self.random_noise_stddev, samples.shape)
		originals = np.asarray(originals)
		return samples, originals

	def get_validation_batch(self):
		if self.validation_data is not None:
			samples = []
			originals = []
			num_sources = len(self.validation_sources)
			sample_shape = self.validation_data[0][0].shape
			for _ in range(self.batch_size):
				waves = []
				sample = np.zeros(sample_shape)
				for s in range(num_sources):
					i = np.random.randint(len(self.validation_data[s]))
					wave = self.validation_data[s][i]
					waves.append(wave)
					sample += wave
				sample = sample/num_sources
				samples.append(sample)
				originals.append(waves)

			samples = np.asarray(samples)
			originals = np.asarray(originals)
			return samples, originals
		else:
			return self.get_training_batch(add_noise=False)


	def normalize_batch(self, batch):
		x = batch.astype(np.float32)
		return x / np.max(np.abs(x))
