---
layout: post
title:  "Experiment 1"
date:   2018-05-06 +0200
categories: result
excerpt_separator: <!-- more -->
---
Experiment 1

No description file found.



Failed to complete!<!-- more -->


{% highlight python %}
{% raw %}
class config(object):
	# Bsub arguments
	bsub_mainfile = "main.py"
	bsub_processors = 4
	bsub_timeout = "4:00"

	# Epoch and batch config
	batch_size = 128 
	latent_dim = 3
	epochs = 400
	epoch_updates = 100

	# Network structure
	input_s = 2048
	n_ae = 2
	n_conv_layers = 3
	n_deconv_layers = 3
	first_size = input_s // (2 ** n_deconv_layers)
	final_decoder_filter_size = 3

	# Miscellaneous constants
	learning_rate = 1e-3
	time_loss_mult = 0.0005
	kl_loss_mult = 0.00003
	kl_extra_mult = 2
	kl_extra_exponent = 2

	# Functions
	@staticmethod
	def deconv_filter_size(i):
		return (2 * (i + 1)) + 1

	@staticmethod
	def deconv_channel_num(i):
		return 2 ** (config.n_deconv_layers + 3 - i)

	@staticmethod
	def conv_filter_size(i):
		return (2 * (config.n_conv_layers - i)) + 1

	@staticmethod
	def conv_channel_num(i):
		return 2 ** (i + 4)

{% endraw %}
{% endhighlight %}
