"""
ASEN : Remote Sensing Data Analysis
Author: Daniel Lindberg

"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance

# Find the mean value of a list
def mean(some_list):
	if len(some_list) == 0:
		return 0
	return sum(some_list)/len(some_list)
# Finds the city block distance between two matrices of x,y pairs
def cityBlockDist(pairs_1, pairs_2):
	all_x=[x[0] for x in pairs_1]
	all_y=[y[1] for y in pairs_1]
	n = len(all_x)
	sum=0
	for i in range(n):
		for j in range(i+1, n):
			sum += abs(all_x[i]-all_x[j])+abs(all_y[i]-all_y[j])
	return sum


# Read the three data points
data_1 = np.load('AIA20110928_0000_0094.npz')['x']
data_2 = np.load('AIA20110928_0000_0171.npz')['x']
data_3 = np.load('AIA20110928_0000_0211.npz')['x']

# Normalize all of the data points
normalized_1 = preprocessing.normalize(data_1)
normalized_2 = preprocessing.normalize(data_2)
normalized_3 = preprocessing.normalize(data_3)

# Create a scaler meant for standardizing the data
scaler_1 = preprocessing.StandardScaler()
scaler_2 = preprocessing.StandardScaler()
scaler_3 = preprocessing.StandardScaler()

# Standardize the data points
scaled_df_1 = scaler_1.fit_transform(normalized_1)
scaled_df_2 = scaler_2.fit_transform(normalized_2)
scaled_df_3 = scaler_3.fit_transform(normalized_3)

# This is for part 1.1 where it asks for the shape
print scaled_df_1.shape
print scaled_df_2.shape
print scaled_df_3.shape

# Specify the various k means values. each with a unique number of clusters
k_means_1 = KMeans(n_clusters=3)
k_means_2 = KMeans(n_clusters=5)
k_means_3 = KMeans(n_clusters=10)

# Fit the amount of clusters to the data
k_means_1.fit(scaled_df_1)
k_means_2.fit(scaled_df_1)
k_means_3.fit(scaled_df_1)

# Get the labels, to determine which cluster group the data is within
labels_1=k_means_1.predict(scaled_df_1)
labels_2=k_means_2.predict(scaled_df_1)
labels_3=k_means_3.predict(scaled_df_1)

# Obtain the centroid centrers for each of the k means clusters
centroids_1 = k_means_1.cluster_centers_
centroids_2 = k_means_2.cluster_centers_
centroids_3 = k_means_3.cluster_centers_

#Plot all 3 data points that are normalized and standardized

# Next this part is for testing the various clusters
# This is ward method of clustering with the euclidean affinity distances calculated, 3 cluster groups
total_clusters_ward = 3
dendrogram = sch.dendrogram(sch.linkage(scaled_df_1, method='ward'))
hc = AgglomerativeClustering(n_clusters=total_clusters_ward, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(scaled_df_1)

# This is ward method of clustering with the euclidean affinity distances calculated,5 cluster groups
total_clusters_ward_5 = 5
dendrogram_5 = sch.dendrogram(sch.linkage(scaled_df_1, method='ward'))
hc_5 = AgglomerativeClustering(n_clusters=total_clusters_ward_5, affinity='euclidean', linkage='ward')
y_hc_5 = hc_5.fit_predict(scaled_df_1)

# This is complete method of clustering with the euclidean affinity distances calculated, 5 cluster groups
total_clusters_complete_5 = 5
dendrogram_complete_5 = sch.dendrogram(sch.linkage(scaled_df_1, method='complete'))
hc_complete_5 = AgglomerativeClustering(n_clusters=total_clusters_complete_5, affinity='euclidean', linkage='complete')
y_hc_complete_5 = hc_complete_5.fit_predict(scaled_df_1)

# This is average method of clustering with the euclidean affinity distances calculated, 5 cluster groups
total_clusters_average_5 = 5
dendrogram_average_5 = sch.dendrogram(sch.linkage(scaled_df_1, method='average'))
hc_average_5 = AgglomerativeClustering(n_clusters=total_clusters_average_5, affinity='euclidean', linkage='average')
y_hc_average_5 = hc_average_5.fit_predict(scaled_df_1)

# This is weighted method of clustering with the euclidean affinity distances calculated, 5 cluster groups
total_clusters_weighted_5 = 5
dendrogram_weighted_5 = sch.dendrogram(sch.linkage(scaled_df_1, method='weighted'))
hc_weighted_5 = AgglomerativeClustering(n_clusters=total_clusters_weighted_5, affinity='euclidean', linkage='ward')
y_hc_weighted_5 = hc_weighted_5.fit_predict(scaled_df_1)

# This is weighted method of clustering with the manhattan affinity distances calculated, 5 cluster groups
total_clusters_weighted_manhattan_5 = 5
dendrogram_manhattan_5 = sch.dendrogram(sch.linkage(scaled_df_1, method='weighted'))
hc_manhattan_5 = AgglomerativeClustering(n_clusters=total_clusters_weighted_manhattan_5, affinity='manhattan', linkage='complete')
y_hc_manhattan_5 = hc_manhattan_5.fit_predict(scaled_df_1)

# This is a color map, when given a number 0-9, each have a color
colmap = {0:'orange', 1: 'r', 2: 'g', 3: 'b', 4: 'c', 5:'m', 6:'y', 7:'k', 8:'olive', 9:'pink'}

#-------------------------------------------------------------------------------------------
# Part 2.1
# Get the x.y points for the labels into a seperate group
groups_3 = []
for i in range(3):
	groups_3.append([])
for i in range(0,len(labels_1)):
	groups_3[labels_1[i]].append((scaled_df_1[0][i], scaled_df_1[1][i]))

# Get the average x,y point for each cluster group
avg_3=[]
for i in range(3):
	avg_3.append([])
for i in range(3):
	avg_3[i] = mean([x[0] for x in groups_3[i]]), mean([x[1] for x in groups_3[i]])

# Get the max cluster size of all groups
max_cluster_size_3 = 0
for i in range(3):
	if max_cluster_size_3 < len(groups_3[i]):
		max_cluster_size_3 = len(groups_3[i])

# Make sure that each cluster has as many points as the max group
for i in range(3):
	while len(groups_3[i]) < max_cluster_size_3:
		groups_3[i].append(avg_3[i])

"""
This works because Euclidean distance is l2 norm and the default 
value of ord parameter in numpy.linalg.norm is 2.
"""
# Obtain the euclidean and city block values for each clustering group
print "Distances, 3 clusters:"
total_euclidean_3 = 0.0
total_cityblock_3 = 0.0
new_set_3 = set()
for i in range(3):
	for j in range(3):
		if i != j and (i,j) not in new_set_3 and (j,i) not in new_set_3:
			new_set_3.add((i,j))
			temp_euclidean = np.linalg.norm(np.array(groups_3[i])-np.array(groups_3[j]))
			temp_city_block = cityBlockDist(groups_3[i], groups_3[j])
			total_euclidean_3 += temp_euclidean
			total_cityblock_3 += temp_city_block
			print "Differences groups "+str(i+1)+"&"+str(j+1)+": Euclidean:" + str(temp_euclidean) + " CityBlocK:" + str(temp_city_block)			
print "Average-Euclidean:"+str(total_euclidean_3/len(new_set_3))
print "Average-CityBlock:"+str(total_cityblock_3/len(new_set_3))
#-------------------------------------------------------------------------------------------
#Also Part 2.1
groups_5 = []
# Get the x.y points for the labels into a seperate group
for i in range(5):
	groups_5.append([])
for i in range(0,len(labels_2)):
	groups_5[labels_2[i]].append((scaled_df_1[0][i], scaled_df_1[1][i]))

# Get the average x,y point for each cluster group
avg_5=[]
for i in range(5):
	avg_5.append([])
for i in range(5):
	avg_5[i] = mean([x[0] for x in groups_5[i]]), mean([x[1] for x in groups_5[i]])

# Get the max cluster size of all groups
max_cluster_size_5 = 0
for i in range(5):
	if max_cluster_size_5 < len(groups_5[i]):
		max_cluster_size_5 = len(groups_5[i])

# Make sure that each cluster has as many points as the max group
for i in range(5):
	while len(groups_5[i]) < max_cluster_size_5:
		groups_5[i].append(avg_5[i])

"""
This works because Euclidean distance is l2 norm and the default 
value of ord parameter in numpy.linalg.norm is 2.
"""
# Obtain the euclidean and city block values for each clustering group
print "Distances, 5 clusters:"
total_euclidean_5 = 0.0
total_cityblock_5 = 0.0
new_set_5 = set()
for i in range(5):
	for j in range(5):
		if i != j and (i,j) not in new_set_5 and (j,i) not in new_set_5:
			new_set_5.add((i,j))
			temp_euclidean = np.linalg.norm(np.array(groups_5[i])-np.array(groups_5[j]))
			temp_city_block = cityBlockDist(groups_5[i], groups_5[j])
			total_euclidean_5 += temp_euclidean
			total_cityblock_5 += temp_city_block
			print "Differences groups "+str(i+1)+"&"+str(j+1)+": Euclidean:" + str(temp_euclidean) + " CityBlocK:" + str(temp_city_block)	
print "Average-Euclidean:"+str(total_euclidean_5/len(new_set_5))
print "Average-CityBlock:"+str(total_cityblock_5/len(new_set_5))
#-------------------------------------------------------------------------------------------
#Also Part 2.1

groups_10 = []
# Get the x.y points for the labels into a seperate group
for i in range(10):
	groups_10.append([])
for i in range(0,len(labels_3)):
	groups_10[labels_3[i]].append((scaled_df_1[0][i], scaled_df_1[1][i]))

# Get the average x,y point for each cluster group
avg_10=[]
for i in range(10):
	avg_10.append([])
for i in range(10):
	avg_10[i] = mean([x[0] for x in groups_10[i]]), mean([x[1] for x in groups_10[i]])

# Get the max cluster size of all groups
max_cluster_size_10 = 0
for i in range(10):
	if max_cluster_size_10 < len(groups_10[i]):
		max_cluster_size_10 = len(groups_10[i])

# Make sure that each cluster has as many points as the max group
for i in range(10):
	while len(groups_10[i]) < max_cluster_size_10:
		groups_10[i].append(avg_10[i])

"""
This works because Euclidean distance is l2 norm and the default 
value of ord parameter in numpy.linalg.norm is 2.
"""
# Obtain the euclidean and city block values for each clustering group
print "Distances, 10 clusters:"
total_euclidean_10 = 0.0
total_cityblock_10 = 0.0
new_set_10 = set()
for i in range(10):
	for j in range(10):
		if i != j and (i,j) not in new_set_10 and (j,i) not in new_set_10:
			new_set_10.add((i,j))
			temp_euclidean = np.linalg.norm(np.array(groups_10[i])-np.array(groups_10[j]))
			temp_city_block = cityBlockDist(groups_10[i], groups_10[j])
			total_euclidean_10 += temp_euclidean
			total_cityblock_10 += temp_city_block
			print "Differences groups "+str(i+1)+"&"+str(j+1)+": Euclidean:" + str(temp_euclidean) + " CityBlocK:" + str(temp_city_block)	
print "Average-Euclidean:"+str(total_euclidean_10/len(new_set_10))
print "Average-CityBlock:"+str(total_cityblock_10/len(new_set_10))

#-------------------------------------------------------------------------------------------

# part 1.3

def findClosest(avg_list, some_point):
	best_distance = 9999999.0
	index = 0
	diff_list = []
	for i in range(len(avg_list)):
		x_diff = abs(some_point[0] - avg_list[i][0])**2
		y_diff = abs(some_point[1] - avg_list[i][1])**2
		diff_avg = math.sqrt(x_diff + y_diff)
		diff_list.append(diff_avg)
		if diff_avg < best_distance:
			best_distance = diff_avg
			index = i
	return index, diff_list

#section 3
#part 1
goal_clusters = 7
scaled_df_7 = scaled_df_1.copy()
k_means_7 = KMeans(n_clusters=goal_clusters)
# Fit the amount of clusters to the data
k_means_7.fit(scaled_df_7)
# Get the labels, to determine which cluster group the data is within
labels_7=k_means_7.predict(scaled_df_7)
# Obtain the centroid centrers for each of the k means clusters
centroids_7 = k_means_7.cluster_centers_
# Get the max cluster size of all groups
temp_lengths = []
for i in range(goal_clusters):
	temp_lengths.append(sum(labels_7==i))
avg_list = mean(temp_lengths)
in_range = True
for every_len in temp_lengths:
	if abs(avg_list - every_len) > 5:
		in_range = False
# Get the average x,y point for each cluster group
all_7=[]
for i in range(goal_clusters):
	all_7.append([])
for x in range(0, len(labels_7)):
	for i in range(0, goal_clusters):
		if labels_7[x] == i:
			all_7[i].append(scaled_df_7[x])
avgs_7 = []
for i in range(0, goal_clusters):
	all_x=[x[0] for x in all_7[i]]
	all_y=[y[1] for y in all_7[i]]
	avgs_7.append( (mean(all_x), mean(all_y)  ) )
print "Start:",in_range, avg_list, temp_lengths
print scaled_df_7.shape
while (not in_range):
	# Get the average x,y point for each cluster group
	all_7=[]
	for i in range(goal_clusters):
		all_7.append([])
	for i in range(0, goal_clusters):
		for x in range(0, len(labels_7)):
			if labels_7[x] == i:
				all_7[i].append(scaled_df_7[x])
	avgs_7 = []
	for i in range(0, goal_clusters):
		all_x=[x[0] for x in all_7[i]]
		all_y=[y[1] for y in all_7[i]]
		avgs_7.append( (mean(all_x), mean(all_y)  ) )
	for j in range(0, len(labels_7)):
		for i in range(0, goal_clusters):
			if labels_7[j] == i :
				if avg_list - temp_lengths[i] < -2:
					# We have a large number and can subtract
					# Take the difference
					diff_value = (scaled_df_7[i][0], scaled_df_7[i][1])
					new_closest, diff_list = findClosest(avgs_7, diff_value)
					# We already have too big of a list, so move to the net closest
					sorted_diffs = sorted(diff_list)
					not_too_big = avg_list-temp_lengths[new_closest] < -2
					index_to_switch = -1
					while (not not_too_big):
						temp_index = diff_list.index(sorted_diffs[index_to_switch])
						if avg_list - temp_lengths[temp_index] > 2:
							# This group is too small, needs input
							labels_7[j] = temp_index
							temp_lengths[temp_index]+=1
							temp_lengths[i]-=1
							not_too_big = True
						elif index_to_switch <= -7:
							not_too_big = True
						index_to_switch -= 1
					break

	temp_lengths = []
	for i in range(goal_clusters):
		temp_lengths.append(sum(labels_7==i))
	avg_list = mean(temp_lengths)
	in_range = True
	for every_len in temp_lengths:
		if abs(avg_list - every_len) > 5:
			in_range = False
	print "Next:",in_range, avg_list, temp_lengths
	

# Part 2 

def findClosestPair(avg_points, exemption_values):
	# This function findest of the centroids of each cluster, the smallest difference
	# Diffs is the set, making sure that there are no copy of sets
	diffs = set()
	#x-y pair and lowest value are values for the pair with smallest values
	x_y_pair = (0,0)
	lowest_value = 99999999999
	for i in range(len(avg_points)):
		for j in range(len(avg_points)):
			# Make sure that these pairs aren't in exception
			if i not in exemption_values and j not in exemption_values:
				#make sure that the pairs aren't in the diff set already
				if i != j and (i,j) not in diffs and (j,i) not in diffs:
					diffs.add((i,j))
					# Compute the distance, like pythagorean theorem
					diff_x = abs(avg_points[i][0] - avg_points[j][0])**2
					diff_y = abs(avg_points[i][1] - avg_points[j][1])**2
					total_diff = math.sqrt(diff_x+diff_y)
					if total_diff < lowest_value:
						lowest_value = total_diff
						x_y_pair = (i,j)
	return x_y_pair

# This is ward method of clustering with the euclidean affinity distances calculated,5 cluster groups
total_clusters_ward_7 = 7
dendrogram_7 = sch.dendrogram(sch.linkage(scaled_df_1, method='ward'))
hc_7 = AgglomerativeClustering(n_clusters=total_clusters_ward_7, affinity='euclidean', linkage='ward')
y_hc_7 = hc_7.fit_predict(scaled_df_1)

# Part 1.1
plt.figure()
plt.subplot(2,2,1)
plt.title('AIA20110928_0000_0094')
plt.plot(scaled_df_1)
plt.subplot(2,2,2)
plt.title('AIA20110928_0000_0171')
plt.plot(scaled_df_2)
plt.subplot(2,2,3)
plt.title('AIA20110928_0000_0211')
plt.plot(scaled_df_3)

#Part 1.2
plt.figure()
# Simply plot the scatter plot for having different number of cluster groups
plt.subplot(2,2,1)
colors_1 = map(lambda x: colmap[x+1], labels_1)
plt.title('3 bands')
plt.scatter(scaled_df_1[0], scaled_df_1[1], color=colors_1, alpha=0.5, edgecolor='k')

plt.subplot(2,2,2)
colors_2 = map(lambda x: colmap[x+1], labels_2)
plt.title('5 bands')
plt.scatter(scaled_df_1[0], scaled_df_1[1], color=colors_2, alpha=0.5, edgecolor='k')

plt.subplot(2,2,3)
colors_3 = map(lambda x: colmap[x], labels_3)
plt.title('10 bands')
plt.scatter(scaled_df_1[0], scaled_df_1[1], color=colors_3, alpha=0.5, edgecolor='k')

# Plot for 1.3
# Below plot the data points for each clustering group
plt.figure()
plt.subplot(3,2,1)
plt.title('Ward-3')
for i in range(0, total_clusters_ward):
	plt.scatter(scaled_df_1[y_hc == i,0], scaled_df_1[y_hc==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
plt.subplot(3,2,2)
plt.title('Ward-5')
for i in range(0, total_clusters_ward_5):
	plt.scatter(scaled_df_1[y_hc_5 == i,0], scaled_df_1[y_hc_5==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
plt.subplot(3,2,3)
plt.title('Complete-5')
for i in range(0, total_clusters_complete_5):
	plt.scatter(scaled_df_1[y_hc_complete_5 == i,0], scaled_df_1[y_hc_complete_5==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
plt.subplot(3,2,4)
plt.title('Average-5')
for i in range(0, total_clusters_average_5):
	plt.scatter(scaled_df_1[y_hc_average_5 == i,0], scaled_df_1[y_hc_average_5==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
plt.subplot(3,2,5)
plt.title('Weighted-5')
for i in range(0, total_clusters_weighted_5):
	plt.scatter(scaled_df_1[y_hc_weighted_5 == i,0], scaled_df_1[y_hc_weighted_5==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
plt.subplot(3,2,6)
plt.title('Weighted-Manhattan-5')
for i in range(0, total_clusters_weighted_manhattan_5):
	plt.scatter(scaled_df_1[y_hc_manhattan_5 == i,0], scaled_df_1[y_hc_manhattan_5==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')

# plot 3.1
plt.figure()
# Simply plot the scatter plot for having different number of cluster groups
colors_7 = map(lambda x: colmap[x+1], labels_7)
plt.title('Part 3.1 Filter')
plt.scatter(scaled_df_7[0], scaled_df_7[1], color=colors_7, alpha=0.5, edgecolor='k')

#Plot 3.2
# First plot the original clustering before we do the refinement
plt.figure()
plt.subplot(2,1,1)
plt.title('Refining Clusters- Begin')
for i in range(0, total_clusters_ward_7):
	plt.scatter(scaled_df_1[y_hc_7 == i,0], scaled_df_1[y_hc_7==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
# Use this to find the centroid for all of the clusters
all_hc_x = []
all_hc_y = []
for i in range(total_clusters_ward_7):
	all_hc_x.append([])
	all_hc_y.append([])
# Add all of the values for a clsuter group together
for i in range(len(y_hc_7)):
	for j in range(total_clusters_ward_7):
		if y_hc_7[i] == j:
			all_hc_x[j].append(scaled_df_1[i][0])
			all_hc_y[j].append(scaled_df_1[i][1])
avgs_hc_7 = []
# Obtain the average or centroid of the cluster
for i in range(total_clusters_ward_7):
	avgs_hc_7.append(( mean(all_hc_x[i]) , mean(all_hc_y[i])  ))
diff_pairs = len(set(y_hc_7))
# Create an exemption set for the merged groups
exempt_set = set()
while(diff_pairs > 2):
	# Find the merge pair
	x_merge, y_merge = findClosestPair(avgs_hc_7, exempt_set)
	# Set all of the merge pairs to one value
	for j in range(len(y_hc_7)):
		if y_hc_7[j]==y_merge:
			y_hc_7[j] = x_merge
	exempt_set.add(y_merge)
	# Do the logic again to get the centroid value
	all_hc_x = []
	all_hc_y = []
	for i in range(total_clusters_ward_7):
			all_hc_x.append([])
			all_hc_y.append([])
	for i in range(len(y_hc_7)):
		for j in range(total_clusters_ward_7):
			if y_hc_7[i] == j:
				all_hc_x[j].append(scaled_df_1[i][0])
				all_hc_y[j].append(scaled_df_1[i][1])
	# Finally determine the average again
	avgs_hc_7 = []
	for i in range(total_clusters_ward_7):
		avgs_hc_7.append(( mean(all_hc_x[i]) , mean(all_hc_y[i])  ))
	# See how many unique values are within the cluster groups
	diff_pairs = len(set(y_hc_7))
plt.subplot(2,1,2)
# Plot the refined cluster after several merges
plt.title('Refining Clusters- Final')
for i in range(0, total_clusters_ward_7):
	plt.scatter(scaled_df_1[y_hc_7 == i,0], scaled_df_1[y_hc_7==i,1],
		s=100, c=colmap[i], alpha=0.5, edgecolor = 'k')
plt.show()
