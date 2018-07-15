import os
import numpy as np
import waveUtils


class config(object):

	def __init__(self):
		self.prepare_data()

	# Bsub arguments
	bsub_mainfile = "main.py"
	bsub_processors = 4
	bsub_timeout = "4:00"
	bsub_memory = 16000

	# Epoch and batch config
	batch_size = 128
	latent_dim = 1000
	epochs = 100
	epoch_updates = 100

	# Network structure
	input_s = 32000
	n_ae = 2

	# Model
	load_model = False
	model_path = os.path.join("models", "0201", "model")  # only used if load_model=True

	# Miscellaneous constants
	sample_rate = 8000
	#learning_rate_min = 1e-4
	#learning_rate_max = 1e-3
	#learning_rate_scaling_factor = -3  # controlls the shape of the scaling curve from max to min learning rate
	learning_rate = 1e-4  # Only works if either of learning_rate_min and learning_rate_max are unspecified
	kl_loss_mult = 1e-4
	kl_extra_mult = 2
	kl_extra_exponent = 2
	useLogProba = False
	useMixtureProba = True
	deterministic = False
	reorder_outputs = False
	var_func = "relu"
	data_sources = ["speech","all_music"]
	data = None

	# Functions
	def prepare_data(self):
		self.load_data()

	def load_and_prepare_audio(self, source, i):
		duration = self.input_s / float(self.sample_rate)
		data_dir = os.path.join("wav_files", source)
		waves, original_sample_rate = waveUtils.loadAudioFiles(data_dir)
		if i == 0:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=6, sample_rate=original_sample_rate, duration=duration)
		else:
			cut_data = waveUtils.extractRandomSegments(waves, num_per_source=1, sample_rate=original_sample_rate, duration=duration)
		del waves
		data = waveUtils.reduceQuality(cut_data, self.sample_rate, duration)
		del cut_data
		return data

	def load_data(self):
		if self.data is None:
			self.data = [self.load_and_prepare_audio(source, i) for i, source in enumerate(self.data_sources)]

	def get_training_batch(self):
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
		originals = np.asarray(originals)
		return samples, originals

	def normalize_batch(self, batch):
		x = batch.astype(np.float32)
		return x / np.max(np.abs(x))
