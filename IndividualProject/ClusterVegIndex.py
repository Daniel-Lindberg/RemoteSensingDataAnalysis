#Author: Daniel Lindberg
# Python Modules
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.cluster.hierarchy as sch
import matplotlib.patches as mpatches

# Native Python Submodules
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from PIL import Image

# Image dimensions to read
img_width_start = 1500
img_width_end = 6000
img_height_start = 1250
img_height_end = 6500

# This is a color map, when given a number 0-9, each have a color
colmap = {0:'r', 1: 'g', 2: 'b', 3: 'm', 4: 'c'}
# Testing image
#img_to_print = "test/LC08_L1TP_026031_20190724_20190801_01_T1/LC08_L1TP_026031_20190724_20190801_01_T1"
		 
# mat directory
mat_dir = "veg_index/"
all_mats = sorted(os.listdir(mat_dir))

# test directory
test_dir = "test/"
all_tests = sorted(os.listdir(test_dir))

all_ndvi_indices = None
all_savi_indices = None
all_msavi_indices = None
"Start loading all mat files"
for sub_mat in all_mats:
	veg_indices = scipy.io.loadmat(mat_dir+sub_mat)
	tmp_ndvi = veg_indices['ndvi']
	tmp_savi = veg_indices['savi']
	tmp_msavi = veg_indices['msavi']
	if not "any" in dir(all_ndvi_indices):
		all_ndvi_indices = tmp_ndvi[550:600]
	else:
		all_ndvi_indices = np.append(all_ndvi_indices, tmp_ndvi[550:600])
		#all_ndvi_indices = np.concatenate((all_ndvi_indices, tmp_ndvi[550:600]))
	if not "any" in dir(all_savi_indices):
		all_savi_indices = tmp_savi[550:600]
	else:
		all_savi_indices = np.append(all_savi_indices, tmp_savi[550:600])		
		#all_savi_indices = np.concatenate((all_savi_indices, tmp_savi[550:600]))
	if not "any" in dir(all_msavi_indices):
		all_msavi_indices = tmp_msavi[550:600]
	else:
		all_msavi_indices = np.append(all_msavi_indices, tmp_msavi[550:600])		
		#all_msavi_indices = np.concatenate((all_msavi_indices, tmp_msavi[550:600]))
	'''	
	for i in range(tmp_ndvi.shape[0]):
		for j in range(tmp_ndvi.shape[1]):
			all_ndvi_indices.append(tmp_ndvi[i,j])
			all_savi_indices.append(tmp_savi[i,j])
			all_msavi_indices.append(tmp_msavi[i,j])
	'''
	print "Finished Loading:", sub_mat, " Current Shape:", all_ndvi_indices.shape
print "Finished loading all .mat files"
# Convert the indices to np arrays
all_ndvi_indices = np.array(all_ndvi_indices)
all_savi_indices = np.array(all_savi_indices)
all_msavi_indices = np.array(all_msavi_indices)

# Let's get rid of all outliers
for x in range(len(all_ndvi_indices)):
	if all_ndvi_indices[x] > 5.0:
		all_ndvi_indices[x] = 5.0
	elif all_ndvi_indices[x] < -5.0:
		all_ndvi_indices[x] = -5.0
	if all_savi_indices[x] > 5.0:
		all_savi_indices[x] = 5.0
	elif all_savi_indices[x] < -5.0:
		all_savi_indices[x] = -5.0
	if all_msavi_indices[x] > 5.0:
		all_msavi_indices[x] = 5.0
	elif all_msavi_indices[x] < -5.0:
		all_msavi_indices[x] = -5.0

# Need to reshape to one feature
all_ndvi_indices = np.reshape(all_ndvi_indices, (-1, 1))
all_savi_indices = np.reshape(all_savi_indices, (-1, 1))
all_msavi_indices = np.reshape(all_msavi_indices, (-1, 1))

print "Creating K-Means Clusters"
# Specify the various k means values. 
k_means_ndvi = KMeans(n_clusters=3, algorithm="elkan")
k_means_savi = KMeans(n_clusters=3, algorithm="elkan")
k_means_msavi = KMeans(n_clusters=3, algorithm="elkan")

# Fit the amount of clusters to the data
k_means_ndvi.fit(all_ndvi_indices)
k_means_savi.fit(all_savi_indices)
k_means_msavi.fit(all_msavi_indices)

print "Finished K-Means Training"
print "Now doing K means prediction for test(s)"

for i in range(len(all_tests)):
	# get last test name
	print "Starting Test:", all_tests[i]
	last_test_name = all_tests[i].split("_VI.")[0]
	test_indices = scipy.io.loadmat(mat_dir+last_test_name+"_VI.mat")
	test_ndvi = np.array(test_indices['ndvi'])
	test_savi = np.array(test_indices['savi'])
	test_msavi = np.array(test_indices['msavi'])
	# Get the shape so that we can reconstruct it later
	test_shape = test_ndvi.shape

	# Shorten outliers
	for x in range(len(all_ndvi_indices)):
		if all_ndvi_indices[x] > 5.0:
			all_ndvi_indices[x] = 5.0
		elif all_ndvi_indices[x] < -5.0:
			all_ndvi_indices[x] = -5.0
		if all_savi_indices[x] > 5.0:
			all_savi_indices[x] = 5.0
		elif all_savi_indices[x] < -5.0:
			all_savi_indices[x] = -5.0
		if all_msavi_indices[x] > 5.0:
			all_msavi_indices[x] = 5.0
		elif all_msavi_indices[x] < -5.0:
			all_msavi_indices[x] = -5.0
	# Shape into our liking
	test_ndvi = np.reshape(test_ndvi, (-1,1))
	test_savi = np.reshape(test_savi, (-1,1))
	test_msavi = np.reshape(test_msavi, (-1,1))

	# Get the labels, to determine which cluster group the data is within
	labels_ndvi=k_means_ndvi.predict(test_ndvi)
	labels_savi=k_means_savi.predict(test_savi)
	labels_msavi=k_means_msavi.predict(test_msavi)

	# There are technically 3 clusters, I want to find which group has highest average cluster
	# So create a list of 3 to be the sum of each cluster
	sum_ndvi = [0.0,0.0,0.0]
	sum_savi = [0.0,0.0,0.0]
	sum_msavi = [0.0,0.0,0.0]
	# Also get a number of how many are in each cluster
	number_ndvi = [0.0, 0.0, 0.0]
	number_savi = [0.0, 0.0, 0.0]
	number_msavi = [0.0, 0.0, 0.0]

	# Reshape back into it's original shape
	test_ndvi = np.reshape(test_ndvi, (test_shape[0], test_shape[1]))
	labels_ndvi = np.reshape(labels_ndvi, (test_shape[0], test_shape[1]))
	test_savi = np.reshape(test_savi, (test_shape[0], test_shape[1]))
	labels_savi = np.reshape(labels_savi, (test_shape[0], test_shape[1]))
	test_msavi = np.reshape(test_msavi, (test_shape[0], test_shape[1]))
	labels_msavi = np.reshape(labels_msavi, (test_shape[0], test_shape[1]))


	for x in range(test_ndvi.shape[0]):
		for y in range(test_ndvi.shape[1]):
			sum_ndvi[labels_ndvi[x,y]] += test_ndvi[x,y]
			sum_savi[labels_savi[x,y]] += test_savi[x,y]
			sum_msavi[labels_msavi[x,y]] += test_msavi[x,y]
			number_ndvi[labels_ndvi[x,y]] += 1.0
			number_savi[labels_savi[x,y]] += 1.0
			number_msavi[labels_msavi[x,y]] += 1.0

	avgs_ndvi = [sum_ndvi[0]/number_ndvi[0], sum_ndvi[1]/number_ndvi[1], sum_ndvi[2]/number_ndvi[2]]
	avgs_savi = [sum_savi[0]/number_savi[0], sum_savi[1]/number_savi[1], sum_savi[2]/number_savi[2]]
	avgs_msavi = [sum_msavi[0]/number_msavi[0], sum_msavi[1]/number_msavi[1], sum_msavi[2]/number_msavi[2]]
	for i in range(3):
		print "Avg-NDVI-"+str(i), ":", avgs_ndvi[i]
		print "Avg-SAVI-"+str(i), ":", avgs_savi[i]
		print "Avg-MSAVI-"+str(i), ":", avgs_msavi[i]

	ndvi_max_index = avgs_ndvi.index(max(avgs_ndvi))
	savi_max_index = avgs_savi.index(max(avgs_savi))
	msavi_max_index = avgs_msavi.index(max(avgs_msavi))
	print "Max-NDVI-Cluster-Group:", ndvi_max_index
	print "Max-SAVI-Cluster-Group:", savi_max_index
	print "Max-MSAVI-Cluster-Group:", msavi_max_index

	print "Size of NDVI Groups:", number_ndvi
	print "Size of SAVI Groups:", number_savi
	print "Size of MSAVI Groups:", number_msavi

	# Obtain the centroid centrers for each of the k means clusters
	centroids_ndvi = k_means_ndvi.cluster_centers_
	centroids_savi = k_means_savi.cluster_centers_
	centroids_msavi = k_means_msavi.cluster_centers_

	print "Finished creating K means cluster for indices"

	# Now to plot how this looks, first open the image
	img_to_print = test_dir+last_test_name+os.sep+last_test_name
	blue_band = np.array(Image.open(img_to_print +"_B2.TIF"))[img_width_start:img_width_end, img_height_start:img_height_end]
	green_band = np.array(Image.open(img_to_print +"_B3.TIF"))[img_width_start:img_width_end, img_height_start:img_height_end]
	red_band = np.array(Image.open(img_to_print +"_B4.TIF"))[img_width_start:img_width_end, img_height_start:img_height_end]


	# Create a copy to be printed

	ndvi_copy = blue_band.copy() 
	savi_copy = blue_band.copy() 
	msavi_copy = blue_band.copy() 
	combinator = blue_band.copy()
	for x in range(red_band.shape[0]):
		for y in range(red_band.shape[1]):
			avg_pixel = sum([red_band[x,y], blue_band[x,y], green_band[x,y]])/3
			if avg_pixel < 150:
				avg_pixel = 150
			is_ndvi_best = labels_ndvi[x,y] == ndvi_max_index
			is_savi_best = labels_savi[x,y] == savi_max_index
			is_msavi_best = labels_msavi[x,y] == msavi_max_index
			if is_ndvi_best:
				# make the image to be green
				ndvi_copy[x,y] = 255 
			else:
				ndvi_copy[x,y] = 0
			if is_savi_best:
				# make the image to be green
				savi_copy[x,y] = 255
			else:
				savi_copy[x,y] = 0
			if is_msavi_best:
				# make the image to be green
				msavi_copy[x,y] = 255
			else:
				msavi_copy[x,y] = 0
			if is_ndvi_best and is_savi_best and is_msavi_best:
				combinator[x,y] = 255
			else:
				combinator[x,y] = 0
	ndvi_copy = Image.fromarray(ndvi_copy)
	savi_copy = Image.fromarray(savi_copy)
	msavi_copy = Image.fromarray(msavi_copy)
	combinator = Image.fromarray(combinator)
	plt.figure()
	plt.title("NDVI Cluster")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(ndvi_copy)
	yellow_patch = mpatches.Patch(color='yellow', label='High Vegetation Index')
	purple_patch = mpatches.Patch(color='purple', label='Low Vegetation Index')
	plt.legend(handles=[yellow_patch, purple_patch])
	plt.savefig(test_dir+last_test_name+os.sep+"NDVI-Cluster.png")
	plt.figure()
	plt.title("SAVI Cluster")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(savi_copy)
	yellow_patch = mpatches.Patch(color='yellow', label='High Vegetation Index')
	purple_patch = mpatches.Patch(color='purple', label='Low Vegetation Index')
	plt.legend(handles=[yellow_patch, purple_patch])
	plt.savefig(test_dir+last_test_name+os.sep+"SAVI-Cluster.png")
	plt.figure()
	plt.title("MSAVI Cluster")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(msavi_copy)
	yellow_patch = mpatches.Patch(color='yellow', label='High Vegetation Index')
	purple_patch = mpatches.Patch(color='purple', label='Low Vegetation Index')
	plt.legend(handles=[yellow_patch, purple_patch])
	plt.savefig(test_dir+last_test_name+os.sep+"MSAVI-Cluster.png")
	plt.figure()
	plt.title("Combinator Cluster")
	plt.xlabel('Pixels X-Direction')
	plt.ylabel('Pixels Y-Direction')
	plt.imshow(combinator)
	yellow_patch = mpatches.Patch(color='yellow', label='High Vegetation Index')
	purple_patch = mpatches.Patch(color='purple', label='Low Vegetation Index')
	plt.legend(handles=[yellow_patch, purple_patch])
	plt.savefig(test_dir+last_test_name+os.sep+"Combinator-Cluster.png")
	#plt.show()
	print "Finished Test:", all_tests[i]


