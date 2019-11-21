#Author: Daniel Lindberg
# Native python modules
import scipy.io
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Native python sobmodules
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC

#Load the .mat files
mat_x = scipy.io.loadmat('HMI20110928_0000_bx.mat')['Bx']
mat_y = scipy.io.loadmat('HMI20110928_0000_by.mat')['By']
mat_z = scipy.io.loadmat('HMI20110928_0000_bz.mat')['Bz']

labeled_mat = scipy.io.loadmat('labeled_SDO_data_HW3.mat')['labeled_data']

# Normalize all of the data points
normalized_x = preprocessing.normalize(mat_x)
normalized_y = preprocessing.normalize(mat_y)
normalized_z = preprocessing.normalize(mat_z)

# Create a scaler meant for standardizing the data
scaler_1 = preprocessing.StandardScaler()
scaler_2 = preprocessing.StandardScaler()
scaler_3 = preprocessing.StandardScaler()

# Standardize the data points
scaled_df_x = scaler_1.fit_transform(normalized_x)
scaled_df_y = scaler_2.fit_transform(normalized_y)
scaled_df_z = scaler_3.fit_transform(normalized_z)

plt.figure()
plt.subplot(3,1,1)
plt.title('Normalized and scaled bx')
plt.plot(scaled_df_x)
plt.subplot(3,1,2)
plt.title('Normalized and scaled by')
plt.plot(scaled_df_y)
plt.subplot(3,1,3)
plt.title('Normalized and scaled bz')
plt.plot(scaled_df_z)


triples = np.zeros((len(scaled_df_x), len(scaled_df_x), 3))
triple_array = []
labeled_array = []
for i in range(len(scaled_df_x)):
	for j in range(len(scaled_df_x)):
		triples[i,j,0] = scaled_df_x[i][j]
		triples[i,j,1] = scaled_df_y[i][j]
		triples[i,j,2] = scaled_df_z[i][j]
		triple_array.append([scaled_df_x[i][j], scaled_df_y[i][j], scaled_df_z[i][j]])
		labeled_array.append(labeled_mat[i][j])
clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(triple_array, labeled_array)
plt.figure()
score = clf.score(triple_array, labeled_array)
params = clf.get_params()
print "Accuracy:",score
print "coef:", clf.coef_
print "Covariance matrix:", clf.covariance_
print "Explained Variance Ratio:", clf.explained_variance_ratio_
print "Means:", clf.means_
print params

pair_set = set()
for i in range(len(clf.coef_)): 
	for j in range(len(clf.coef_)):
		if (i,j) not in pair_set and (j,i) not in pair_set and i!=j:
			pair_set.add((i,j))
			x_difference = (clf.coef_[i][0] - clf.coef_[j][0])**2
			y_difference = (clf.coef_[i][1] - clf.coef_[j][1])**2
			z_difference = (clf.coef_[i][2] - clf.coef_[j][2])**2
			print "Difference, between:"+str(i)+' and ' + str(j) + ":" + str(x_difference+y_difference+z_difference)
print "Accuracy, first 100:", clf.score(triple_array[1:100], labeled_array[1:100])
print "Accuracy, first 200:",clf.score(triple_array[1:200], labeled_array[1:200])
print "Accuracy, second 100:",clf.score(triple_array[100:200], labeled_array[100:200])
print "Accuracy, 200-210:",clf.score(triple_array[200:210], labeled_array[200:210])


rbf_svm_model = SVC()
rbf_svm_model.fit(triple_array, labeled_array)
print "Accuracy:",rbf_svm_model.score(triple_array, labeled_array)
print "support:",rbf_svm_model.n_support_
print "params:",rbf_svm_model.get_params()
rbf_Z = rbf_svm_model.predict(triple_array)
linear_svm_model = SVC(kernel="linear")
linear_svm_model.fit(triple_array, labeled_array)
print "Accuracy:",linear_svm_model.score(triple_array, labeled_array)
print "coef:",linear_svm_model.coef_
print "support:",linear_svm_model.n_support_
print "params:",linear_svm_model.get_params()
linear_Z = linear_svm_model.predict(triple_array)
poly_svm_model = SVC(kernel="poly")
poly_svm_model.fit(triple_array, labeled_array)
print "Accuracy:",poly_svm_model.score(triple_array, labeled_array)
print "support:",poly_svm_model.n_support_
print "params:",poly_svm_model.get_params()
poly_Z = poly_svm_model.predict(triple_array)

plt.figure()
plt.subplot(3,1,1)
plt.title("RBF Model")
plt.plot(rbf_Z)
plt.subplot(3,1,2)
plt.title("linear Model")
plt.plot(linear_Z)
plt.subplot(3,1,3)
plt.title("Poly Model")
plt.plot(poly_Z)



# Part 3


plt.show()
