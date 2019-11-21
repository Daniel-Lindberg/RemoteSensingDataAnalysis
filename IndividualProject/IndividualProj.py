#Author: Daniel Lindberg
# Native python modules
import cv2
import os
import scipy.io
import scipy.misc
#import skimage
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
#Native python submodules
#from skimage import io
from PIL import Image

imgs_dir = "imgs/"

nir_photos = []
for root, dirs, files in os.walk(imgs_dir):
	for name in files:
		if "_B5.TIF" in name:
			nir_photos.append(os.path.join(root, name))

for nir_p in nir_photos:
	file_name = nir_p.split(os.sep)[-1]
	im = cv2.imread(nir_p)
	image_sum = sum(im)
	print "File:",file_name, " Median:", im[3000] ," sum:", image_sum
#im_1 = np.asarray(Image.open(imgs_directory+"LC08_L1TP_019033_20190925_20191017_01_T1_QB.tif"))
"""im_2 = cv2.imread(imgs_directory+"LC08_L1TP_019033_20190925_20191017_01_T1.tif")
im_3 = Image.open(imgs_directory+"LC08_L1TP_019033_20190925_20191017_01_T1_TIR.tif")
im_3_2 = plt.imread(imgs_directory+"LC08_L1TP_019033_20190925_20191017_01_T1_TIR.tif")
im_3_3 = cv2.imread(imgs_directory+"LC08_L1TP_019033_20190925_20191017_01_T1_TIR.tif")
im_t1=cv2.imread(imgs_directory+"LC08_L1TP_024031_20190827_20190903_01_T1_B1.TIF")
im_t2=cv2.imread(imgs_directory+"LC08_L1TP_024031_20190827_20190903_01_T1_B2.TIF")
im_t3=cv2.imread(imgs_directory+"LC08_L1TP_024031_20190827_20190903_01_T1_B3.TIF")
"""
#print im_1[3000][3000]
#print im_1.shape
#print im_2.shape
print im_3
print dir(im_3)
print im_3_3[3000][3000]
print im_2[3000][3000]
print im_t1[3000][3000]
print im_t2[3000][3000]
print im_t3[3000][3000]
plt.figure()
plt.subplot(2,1,1)
plt.imshow(im_3_3)
plt.subplot(2,1,2)
plt.imshow(im_2)
plt.show()
"""print im_3.getpixel((3000,3000))
print im_3.getpixel((3000,3000,1))
print im_3.getpixel((3000,3000,2))
print im_3.getpixel((3000,3000,3))
print im_3.getpixel((3000,3000,4))"""