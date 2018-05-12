"""
	Created by Qiang at 16/04/2018
"""


# the data used for training and testing
data_used = 'all'  # native, postcontrast, all
score = 'motion'  # ''motion' poorquality # the score to detected / classify

if score is 'motion':
	load_pha = False
	use_pha = False
	load_r2map = True
	use_r2map = False
	use_t1map = False
	use_mxmask = False  # apply myocardium mask to t1 maps before classification
	# sh7_only = True  # disgard t1map and r2map in the cnn training

	normalise = True
	# image_size = (160, 160)
	image_size = (160, 160)
	image_depth = 7
	datapath = 'C://Users\qiangz\Documents//180101_HCMR_artefacts\gradcam_scoring_active\data//motion//'
	nchunk = 1  # number of chunk data devided for training
	train_from_previous = False

if score is 'poorquality':
	load_pha = False
	use_pha = False
	load_r2map = True
	use_r2map = True

	use_mxmask = False  # apply myocardium mask to t1 maps before classification

	normalise = True
	image_size = (160, 160)
	# image_size = (224, 224)
	image_depth = 7
	datapath = 'data//poorquality//'
	nchunk = 2  # number of chunk data devided for training



