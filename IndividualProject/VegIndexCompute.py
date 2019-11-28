#Author: Daniel Lindberg
# Native python modules
#import cv2
import math
import os
import scipy.io
#import scipy.misc
#import skimage
import numpy as np
#import matplotlib.pyplot as plt
import tifffile as tiff
#Native python submodules
#from skimage import io
from PIL import Image
from sklearn import preprocessing

imgs_dir = "test/"
output_dir = "veg_index/"
individual_imgs = os.listdir(imgs_dir)

def normalizeAndStandardize(some_np_array):
	normalized_data = preprocessing.normalize(some_np_array)
	scaler_data = preprocessing.StandardScaler()
	scaled_data = scaler_data.fit_transform(normalized_data)
	return scaled_data

"""
nir_photos = []
for root, dirs, files in os.walk(imgs_dir):
	for name in files:
		if "_B5.TIF" in name or "_B4.TIF" in name:
			nir_photos.append(os.path.join(root, name))
"""
img_width_start = 1500
img_width_end = 6000
img_height_start = 1250
img_height_end = 6500
last_img = None
for sub_dir in individual_imgs:
	band_4 = imgs_dir+sub_dir+os.sep+sub_dir+"_B4.TIF"
	band_5 = imgs_dir+sub_dir+os.sep+sub_dir+"_B5.TIF"
	# Get all of the bands into a numpy array
	tif_im_4 = Image.open(band_4)
	tif_im_4 = np.array(tif_im_4, dtype=np.float64)
	tif_im_5 = Image.open(band_5)
	tif_im_5 = np.array(tif_im_5, dtype=np.float64)
	# Typically the geo images have diagonols, so we want to crop the points
	tif_im_4 = tif_im_4[img_width_start:img_width_end, img_height_start:img_height_end]
	tif_im_5 = tif_im_5[img_width_start:img_width_end, img_height_start:img_height_end]
	# now normalize and standardize
	tif_im_4 = normalizeAndStandardize(tif_im_4)
	tif_im_5 = normalizeAndStandardize(tif_im_5)
	ndvi_index = np.zeros((tif_im_4.shape[0], tif_im_4.shape[1]))
	msavi_index = np.zeros((tif_im_4.shape[0], tif_im_4.shape[1]))
	savi_index = np.zeros((tif_im_4.shape[0], tif_im_4.shape[1]))
	for x in range(tif_im_4.shape[0]):
		for y in range(tif_im_4.shape[1]):
			ndvi_index[x,y] = (tif_im_5[x,y] - tif_im_4[x,y])/(tif_im_5[x,y] + tif_im_4[x,y])
			msavi_index[x,y] = (2.0 * tif_im_5[x,y] + 1 - math.sqrt(abs((2.0* tif_im_5[x,y] + 1.0)**2 - 8.0 * (tif_im_5[x,y]-tif_im_4[x,y]))))/2.0
			savi_index[x,y] = ((tif_im_5[x,y] - tif_im_4[x,y])/(tif_im_5[x,y] + tif_im_4[x,y] + 0.5)) * 1.5
	
	scipy.io.savemat(output_dir+sub_dir+'_VI.mat', {'ndvi': ndvi_index, 'savi': savi_index, 'msavi': msavi_index})	
	

