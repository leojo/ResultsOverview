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

	# Miscellaneous constants
	min_freq = 100
	max_freq = 800
	num_freqs = 50
	target_sample_rate = 8000
	sample_rate = 8000
	validation_samples = 10
	_val_data = None
	_val_originals = None
	time_loss_mult = 0.0005
	reconstruction_mult = 1
	learning_rate = 1e-3
	kl_loss_mult = 2e-4
	kl_extra_mult = 2
	kl_extra_exponent = 4
	use_square = False
	data_source = ["chello"]

	# Functions
	def get_training_data(self):
		all_data = []
		all_originals = []
		if len(self.data_source) == 1:
			if self.data_source[0] == "wave":
				waves = waveUtils.generateRandomFrequencies(self.input_s, num=self.num_freqs, freq_min=self.min_freq,
				                                            freq_max=self.max_freq, num_to_combine=2)
			else:
				waves, original_sample_rate = waveUtils.loadAudioFiles(self.data_source[0])
				sample_factor = (original_sample_rate // self.target_sample_rate)
				self.sample_rate = original_sample_rate / sample_factor
				waves = waveUtils.reduceQuality(waves, sample_factor)
				waves = waveUtils.extractCenters(waves, num_samples=self.input_s, latest_start=self.sample_rate)
			for i in range(len(waves)):
				for j in range(i+1, len(waves)):
					all_data.append(waves[i] + waves[j])
					all_originals.append([waves[i], waves[j]])
		elif len(self.data_source) == 2:
			waves1, original_sample_rate1 = waveUtils.loadAudioFiles(self.data_source[0])
			sample_factor1 = (original_sample_rate1 // self.target_sample_rate)
			self.sample_rate = original_sample_rate1 / sample_factor1
			waves1 = waveUtils.reduceQuality(waves1, sample_factor1)
			waves1 = waveUtils.extractCenters(waves1, num_samples=self.input_s, latest_start=self.sample_rate)

			waves2, original_sample_rate2 = waveUtils.loadAudioFiles(self.data_source[1])
			sample_factor2 = (original_sample_rate2 // self.target_sample_rate)
			if self.sample_rate != original_sample_rate2 / sample_factor2:
				print "Mismatching sample rates!"
				exit(1)
			waves2 = waveUtils.reduceQuality(waves2, sample_factor2)
			waves2 = waveUtils.extractCenters(waves2, num_samples=self.input_s, latest_start=self.sample_rate)

			for i in range(len(waves1)):
				for j in range(len(waves2)):
					all_data.append(waves1[i] + waves2[j])
					all_originals.append([waves1[i], waves2[j]])
		else:
			print("More than one data sources are not supported yet")
			exit(1)

		num_samples = len(all_data)
		perm = np.random.permutation(num_samples)
		all_data = np.asarray(all_data)[perm]
		all_originals = np.asarray(all_originals)[perm]
		all_data = np.asarray(all_data)
		all_data = all_data + np.random.normal(scale=self.random_noise_stdev, size=all_data.shape)
		self._val_data = all_data[:self.validation_samples]
		self._val_originals = all_originals[:self.validation_samples]
		return all_data[self.validation_samples:], all_originals[self.validation_samples:]

	def get_validation_data(self):
		if self.validation_samples is None:
			self.get_training_data()
		data = []
		originals = []
		for i in range(self.batch_size):
			idx = min(i, len(self._val_data) - 1)
			data.append(self._val_data[idx])
			originals.append(self._val_originals[idx])
		return np.asarray(data), np.asarray(originals)

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
