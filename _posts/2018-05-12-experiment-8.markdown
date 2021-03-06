---
layout: post
title:  "Experiment 8"
date:   2018-05-12 +0200
categories: result
excerpt_separator: <!-- more -->
---
Experiment 8

No description file found.



Loss | Reconstruction | KL | Completion | Epochs | Elapsed | Remaining | Speed
0.0690 | 0.0648 | 13992.3648 | 100% | 200/200 | 09:16 | 00:00 | 2.78s/it<!-- more -->

## **Sample batch**:
_sample plots_:
![sample_plots]({{"/results/0008/sample_plots.png"| absolute_url}}){:width="1000px"}



{% highlight python %}
{% raw %}
import numpy as np
import waveUtils


class config(object):
	# Bsub arguments
	bsub_mainfile = "main.py"
	bsub_processors = 4
	bsub_timeout = "4:00"

	# Epoch and batch config
	batch_size = 128 
	latent_dim = 3
	epochs = 200
	epoch_updates = 100

	# Network structure
	input_s = 2048
	n_ae = 2
	n_conv_layers = 3
	n_deconv_layers = 3
	first_size = input_s // (2 ** n_deconv_layers)
	final_decoder_filter_size = 3

	# Miscellaneous constants
	min_freq = 100
	max_freq = 800
	num_freqs = 300
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
		waves = waveUtils.generateRandomFrequencies(self.input_s, self.num_freqs, self.min_freq, self.max_freq)
		all_data = []
		all_originals = []
		for i in range(len(waves)):
			for j in range(i,len(waves)):
				all_data.append(waves[i]+waves[j])
				all_originals.append([waves[i],waves[j]])
		self._val_data = np.asarray(all_data[:self.validation_samples])
		self._val_originals = np.asarray(all_originals[:self.validation_samples])
		return np.asarray(all_data[self.validation_samples:]), np.asarray(all_originals[self.validation_samples:])

	def get_validation_data(self):
		data = []
		originals = []
		for i in range(self.batch_size):
			idx = min(i,len(self._val_data))
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

{% endraw %}
{% endhighlight %}
