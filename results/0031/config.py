import numpy as np
import waveUtils


class config(object):
	# Bsub arguments
	bsub_mainfile = "myMain.py"
	bsub_processors = 4
	bsub_timeout = "4:00"

	# Epoch and batch config
	batch_size = 128
	latent_dim = 100
	epochs = 400
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
	learning_rate = 1e-3
	time_loss_mult = 0.0005
	kl_loss_mult = 0.00003
	kl_extra_mult = 2
	kl_extra_exponent = 2

	# Functions
	def get_training_data(self):
		# waves = waveUtils.generateRandomFrequencies(self.input_s, num=self.num_freqs, freq_min=self.min_freq,
		#                                            freq_max=self.max_freq, num_to_combine=4,
		#                                            samples_per_sec=self.sample_rate)

		waves1, original_sample_rate1 = waveUtils.loadAudioFiles("chello")
		sample_factor1 = (original_sample_rate1 // self.target_sample_rate)
		self.sample_rate = original_sample_rate1 / sample_factor1
		waves1 = waveUtils.reduceQuality(waves1, sample_factor1)
		waves1 = waveUtils.extractCenters(waves1, num_samples=self.input_s, latest_start=self.sample_rate)

		#waves2, original_sample_rate2 = waveUtils.loadAudioFiles("chello")
		#sample_factor2 = (original_sample_rate2 // self.target_sample_rate)
		#if self.sample_rate != original_sample_rate2 / sample_factor2:
		#	print "Mismatching sample rates!"
		#	exit(1)
		#waves2 = waveUtils.reduceQuality(waves2, sample_factor2)
		#waves2 = waveUtils.extractCenters(waves2, num_samples=self.input_s, latest_start=self.sample_rate)

		all_data = []
		all_originals = []
		for i in range(len(waves1)):
			for j in range(len(waves1)):
				all_data.append(waves1[i] + waves1[j])
				all_originals.append([waves1[i], waves1[j]])
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

	def deconv_filter_size(self, i):
		return (2 * (i + 1)) + 1

	def deconv_channel_num(self, i):
		return 2 ** (config.n_deconv_layers + 3 - i)

	def conv_filter_size(self, i):
		return (2 * (config.n_conv_layers - i)) + 1

	def conv_channel_num(self, i):
		return 2 ** (i + 4)
