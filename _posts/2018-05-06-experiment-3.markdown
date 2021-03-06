---
layout: post
title:  "Experiment 3"
date:   2018-05-06 +0200
categories: result
tags: bestof
excerpt_separator: <!-- more -->
---
First experiment successfully run on generated audio signal (pure sine wave).

Latent dimension 3. KL loss multiplier 1e-5. No Validation set.



Loss | Reconstruction | KL | Completion | Epochs | Elapsed | Remaining | Speed
0.0040 | 0.0017 | 7876.8561 | 100% | 400/400 | 18:06 | 00:00 | 2.72s/it<!-- more -->

## **Sample batch**:
_sample plots_:
![sample_plots]({{"/results/0003/sample_plots.png"| absolute_url}}){:width="1000px"}



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
