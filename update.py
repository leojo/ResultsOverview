import os
import re
import sys
import glob
from datetime import datetime
import locale
print("Synchronizing results directories. This may take a while...")
os.system("rsync -arvz -e ssh leoj@login.leonhard.ethz.ch:~/sandbox/results .")
print("Updating markdowns")

def process(string):
	match = re.search("[0-9]:", string)
	headers = []
	values = []
	if not match:
		return None

	working_string = string[:match.end()-1]
	for header_val_string in working_string.split(","):
		h, v = header_val_string.strip().split(":")
		h = h.strip()
		v = v.strip()
		headers.append(h)
		values.append(v)

	working_string = string[match.end():].strip()
	headers.append("Completion")
	values.append(working_string.split("|")[0].strip())
	headers.append("Epochs")
	values.append(working_string.split("|")[2].split()[0].strip())
	headers.append("Elapsed")
	values.append(working_string.split("|")[2].split()[1][1:-1].split("<")[0])
	headers.append("Remaining")
	values.append(working_string.split("|")[2].split()[1][1:-1].split("<")[1])
	headers.append("Speed")
	values.append(working_string.split("|")[2].split()[2][:-1])
	header_line = " | ".join(headers)
	value_line = " | ".join(values)
	return header_line+"\n"+value_line

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)



post_template = \
"""---
layout: post
title:  "Experiment {}"
date:   {} +0200
categories: result
excerpt_separator: <!-- more -->
---
{}


{}
"""

exp_dirs = sorted(glob.glob("results/[0-9]*"))
post_files = glob.glob("_posts/*-experiment-*.markdown")
post_path_template = "_posts/{}-experiment-{}.markdown"

#posted_experiments = [int(x.split("-")[-1].split(".")[0]) for x in post_files]

#for path in exp_dirs:
#	exp_num = int(os.path.basename(path))
#	if exp_num not in posted_experiments:
#		print "Experiment {}".format(exp_num)

for exp_dir in exp_dirs:
	print "Creating markdown for {}".format(exp_dir)
	overview = ""
	exp_num = int(os.path.basename(exp_dir))
	desciption_path = os.path.join(exp_dir, "description.txt")
	if os.path.exists(desciption_path):
		with open(desciption_path) as o:
			desciption = o.read().strip().replace("\n","\n\n")
	else:
		desciption = "Experiment {}\n\nNo description file found.".format(exp_num)

	output_path = os.path.join(exp_dir, "output.txt")
	scores = []
	score_headers = []
	date = None
	if not os.path.exists(output_path):
		print("Continuing...")
		continue
	with open(output_path) as o:
		loss_line = None
		lines = o.read().replace("\r","\n").split("\n")
		for line in lines:
			if "Results reported" in line:
				date_line = line.strip().split("at")[-1].strip()
				if date_line[-1] == ".":
					date_line = date_line[:-1]
				try:
					date = datetime.strptime(date_line, "%a %b %d %H:%M:%S %Y")
				except Exception, e:
					date = datetime.strptime(date_line, "%b %d %H:%M:%S %Y")
			if "Loss:" in line:
				loss_line = line.split("\r")[-1].strip()
			if "Average separation score" in line:
				score_headers.append(line.split(":")[-2].split()[-1])
				scores.append(line.split()[-1])
	if loss_line is None:
		loss_line = "Failed to complete!"
	else:
		loss_line = process(loss_line)
	scores_from_wav = False
	if len(score_headers) == 0 or len(scores) == 0:
		wav_sep_score_path = os.path.join(exp_dir,"wav_sep_scores.txt")
		if os.path.exists(wav_sep_score_path):
			with open(os.path.join(exp_dir,"wav_sep_scores.txt")) as o:
				lines = o.readlines()
				score_headers = lines[0].strip().split("|")
				scores = lines[1].split("|")
				scores_from_wav = True
	if date is None:
		print("Unable to parse date for {}".format(exp_dir))
		continue
	date_string = date.strftime("%Y-%m-%d")
	score_line = ""
	if len(scores) > 0:
		score_line = "Separation scores{}:\n\n".format(" (reconstructed)" if scores_from_wav else "")+"|".join(score_headers)+"\n"+"|".join(scores)
	overview = "{}<!-- more -->\n\n{}\n\n{}".format(score_line, desciption, loss_line)
	img_width = 1000
	img_embed_template = "![$D]({{\"/$P\"| absolute_url}}){:width=\"$Wpx\"}"
	audio_embed_template = "<audio src=\"/ResultsOverview/{}\" controls preload></audio>"
	for prefix in ["Sample", "Validation"]:
		glob_string = os.path.join(exp_dir,"{}_*.png".format(prefix.lower()))
		sample_img_paths = natural_sort(glob.glob(glob_string))
		if len(sample_img_paths) > 0:
			overview += "\n\n## **{} batch**:\n".format(prefix)
			if len(sample_img_paths) > 1:
				images = []
				for img_path in sample_img_paths:
					img_description = os.path.basename(img_path)[:-4]
					image_entry = img_embed_template\
						.replace("$D",img_description)\
						.replace("$P",img_path)\
						.replace("$W",str(img_width))
					audio_path = img_path[:-4]+".wav"
					if os.path.exists(audio_path):
						image_entry = audio_embed_template.format(audio_path)+"\n"+image_entry
					image_entry = "_"+img_description.replace("_", " ")+"_:\n"+image_entry
					images.append(image_entry)
				overview += "\n\n".join(images)
			else:
				glob_string = os.path.join(exp_dir,"{}_*.wav".format(prefix.lower()))
				sample_audio_paths = natural_sort(glob.glob(glob_string))
				img_path = sample_img_paths[0]
				img_description = os.path.basename(img_path)[:-4]
				image_entry = img_embed_template\
					.replace("$D",img_description)\
					.replace("$P",img_path)\
					.replace("$W",str(img_width))
				image_entry = "_"+img_description.replace("_", " ")+"_:\n"+image_entry
				overview += image_entry+"\n"
				audios = []
				for audio_path in sample_audio_paths:
					audio_description = os.path.basename(audio_path)[:-4]
					audio_entry = audio_embed_template.format(audio_path)
					audio_entry = "_"+audio_description.replace("_", " ")+"_:\n"+audio_entry
					audios.append(audio_entry)
				overview += "\n\n".join(audios)
	with open(os.path.join(exp_dir,"config.py")) as o:
		config = o.read()
		config = "{% highlight python %}\n{% raw %}\n"+config+"\n{% endraw %}\n{% endhighlight %}"
	with open(post_path_template.format(date_string, exp_num), "w") as post:
		post.write(post_template.format(exp_num, date_string, overview, config))

os.system("git add -A && git commit -m \"Updating Overview\" && git pull origin master --no-edit && git push origin master")