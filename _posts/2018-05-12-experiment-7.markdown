---
layout: post
title:  "Experiment 7"
date:    +0200
categories: result2018-05-12
excerpt_separator: <!-- more -->
---
<!-- more -->

Experiment 7

No description file found.

Loss | Reconstruction | KL | Completion | Epochs | Elapsed | Remaining | Speed
0.0034 | 0.0012 | 7290.2448 | 100% | 600/600 | 27:52 | 00:00 | 2.79s/it

## **Sample batch**:
_sample plots_:
![sample_plots]({{"/results/0007/sample_plots.png"| absolute_url}}){:width="1000px"}


## **Validation batch**:
_validation plots_:
![validation_plots]({{"/results/0007/validation_plots.png"| absolute_url}}){:width="1000px"}



{% highlight python %}
{% raw %}
class config(object):
	# Bsub arguments
	bsub_mainfile = "myMain.py"
	bsub_processors = 4
	bsub_timeout = "4:00"

	# Epoch and batch config
	batch_size = 128 
	latent_dim = 3
	epochs = 600
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
