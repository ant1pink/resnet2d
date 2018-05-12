"""
    Created by Qiang at 30/04/2018
"""
"""
	Created by Qiang at 19/04/2018
"""
import matplotlib.pyplot as plt
import numpy as np

def remove_keymap_conflicts(new_keys_set):
	for prop in plt.rcParams:
		if prop.startswith('keymap.'):
			keys = plt.rcParams[prop]
			remove_list = set(keys) & new_keys_set
			for key in remove_list:
				keys.remove(key)


def imshow3d(volume, title=''):
	volume = np.transpose(volume, (2, 0, 1))
	remove_keymap_conflicts({'j', 'k'})
	fig, ax = plt.subplots()
	ax.volume = volume
	ax.index = volume.shape[0] // 2
	ax.title1 = title
	ax.imshow(volume[ax.index])
	plt.title(ax.title1+': slice'+str(ax.index))
	fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
	fig = event.canvas.figure
	ax = fig.axes[0]
	if event.key == 'j':
		previous_slice(ax)
	elif event.key == 'k':
		next_slice(ax)
	fig.canvas.draw()

def previous_slice(ax):
	volume = ax.volume
	ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
	ax.images[0].set_array(volume[ax.index])
	plt.title(ax.title1+': slice'+str(ax.index))


def next_slice(ax):
	volume = ax.volume
	ax.index = (ax.index + 1) % volume.shape[0]
	ax.images[0].set_array(volume[ax.index])
	plt.title(ax.title1+': slice'+str(ax.index))
