# Author: Daniel Lindberg
# Native python modules
import os
import numpy as np
import matplotlib.pyplot as plt

# Native python submodules
from PIL import Image

# Image dimensions to read
img_width_start = 1500
img_width_end = 6000
img_height_start = 1250
img_height_end = 6500

# Get the image sizes
img_width = img_width_end - img_width_start
img_height = img_height_end - img_height_start

# Img test dir
img_dir = "test/"
sub_dirs = os.listdir(img_dir)
output_image = np.zeros([img_width, img_height, 3], dtype=np.uint8)
for s_d in sub_dirs:
	blue_band = np.array(Image.open(img_dir+s_d+os.sep+s_d+"_B2.TIF"))[img_width_start:img_width_end, img_height_start:img_height_end]
	green_band = np.array(Image.open(img_dir+s_d+os.sep+s_d+"_B3.TIF"))[img_width_start:img_width_end, img_height_start:img_height_end]
	red_band = np.array(Image.open(img_dir+s_d+os.sep+s_d+"_B4.TIF"))[img_width_start:img_width_end, img_height_start:img_height_end]
	for x in range(blue_band.shape[0]):
		for y in range(blue_band.shape[0]):
			output_image[x,y] = [red_band[x,y], green_band[x,y], blue_band[x,y]]
	img = Image.fromarray(output_image)
	img.save('testrgb.png')


	
