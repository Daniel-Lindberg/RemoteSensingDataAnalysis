#Author: Daniel Lindberg
import math
import numpy as np

from sklearn.naive_bayes import GaussianNB

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


# 1 = red , 2= yellow
colors = [1,1,1,2,2,2,2,2,1,1]
# 3 = sports , 4 = SUV
types = [3,3,3,3,3,4,4,4,4,3]
# 5 = Domestic, 6 = Imported
origin = [5,5,5,5,6,6,6,5,6,6]
all_data = []
for i in range(len(colors)):
	all_data.append([colors[i], types[i], origin[i]])
nb_y = [1,0,1,0,1,0,1,0,0,1]

yellow_imported_sport = [2,3,6]

gnb = GaussianNB()
y_fit = gnb.fit(all_data, nb_y)
y_pred= y_fit.predict(all_data)
print "Naive Bayes Problem"
print("Number of mislabeled points out of a total %d points : %d"
       % (len(colors),(nb_y != y_pred).sum()))
print "Probability:",y_fit.class_prior_
print "Training Samples:",y_fit.class_count_
print "Theta:",y_fit.theta_
print "Sigma:",y_fit.sigma_
#print "Epsilon:",y_fit.epsilon_
print "Get Params:",y_fit.get_params
print "Predict:",y_pred
print "Actual:", nb_y
for i in range(len(y_pred)):
	if y_pred[i] != nb_y[i]:
		print "Bad:", i+1
	else:
		print "Good:", i+1
print "-------------------------------------------"
print "Neural Network Problem"


# Input Layers
x1 = [0,0,0,0,1,1,1,1]
x2 = [0,0,1,1,0,0,1,1]
x3 = [0,1,0,1,0,1,0,1]

all_x = []
for i in range(len(x1)):
	all_x.append([x1[i], x2[i], x3[i]])

# Output Layers
y_output = [10, -5 , -5 , 10, -5, 10, 10, 10]

# linear and sign hidden nodes
N_hidden_nodes = 2

all_x = np.array([np.array(xi) for xi in all_x])

w1 = np.random.rand(all_x.shape[1], 3)
w2 = np.random.rand(3,1)

layer1 = sigmoid(np.dot(all_x, w1))
output_layer = sigmoid(np.dot(layer1, w2))

print "Weight1:", w1
print "Weight2:", w2
print "Layer1:",layer1
print "Output:", output_layer

print "-------------------------------------------"
print "Decision Tree Problem"

class_values = [1,2,3]
total_data_points = [6, 3, 1]

total_gini_index = 0.0
for i in range(len(class_values)):
	d_t_formula = total_data_points[i] / float(sum(total_data_points))
	temp_gini_index = d_t_formula * (1.0-d_t_formula)
	total_gini_index += temp_gini_index
print "Gini_index:", total_gini_index

total_split_d1 = [5, 3, 1]
split_d1_gini = 0.0
for i in range(len(total_split_d1)):
	d_t_formula = total_split_d1[i] / float(sum(total_split_d1))
	temp_gini_index = d_t_formula * (1.0-d_t_formula)
	split_d1_gini += temp_gini_index
print "Gini_index [Split d1 at 0.8]:", split_d1_gini


total_split_d2 = [5, 1, 1]
split_d2_gini = 0.0
for i in range(len(total_split_d2)):
	d_t_formula = total_split_d2[i] / float(sum(total_split_d2))
	temp_gini_index = d_t_formula * (1.0-d_t_formula)
	split_d2_gini += temp_gini_index
print "Gini_index [Split d2 at 0.8]:", split_d2_gini