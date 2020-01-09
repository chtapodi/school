'''
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6308584/
https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
'''
#!/usr/bin/env python
# coding: utf-8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Author: Xavier Theo Quinn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
I am going to utilize regression to go from rssi to a distance.
'''
#Imports
import csv
import matplotlib.pyplot as plt
import math
import copy
import random
import statistics as stat
import ast

from string import ascii_lowercase
from datetime import datetime
from random import randint
import itertools
from statistics import mean
from scipy.spatial import distance

import numpy as np
from sklearn import linear_model

import ast
import time
from mpl_toolkits.mplot3d import Axes3D

#SCIKIT-LEARN
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_predict

from sklearn.metrics.scorer import make_scorer

from sklearn.neighbors import KNeighborsClassifier


from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

# And import the feature selection mechanism

from sklearn.feature_selection import RFE
from sklearn.metrics.scorer import make_scorer

from sklearn import preprocessing
from sklearn import utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ##Methods:
# #General
#Format print
# VERBOSE=False
VERBOSE=True
COMPARISON=False


def fprint(label, value) :
	try: VERBOSE
	except NameError: print("{0}: {1}\n".format(label, value))
	else:
		if VERBOSE :
			print("{0}: {1}\n".format(label, value))


#Quick print
def qprint(*args) :
	var=args[len(args)-1]
	for i in range(len(args)-1) :
		print(args[i], end ="")

	print("{0}: {1}".format(var, repr(eval(var)))) #yeah yeah, insecure methods, but I doubt anyone will be tryig to inject code into my ML hw.

#prints out lists of lists in the format most commonly found in this data.
def lprint(label,toprint) :
	# print(label)
	for data in toprint :
		# for variableName in data :
		fprint(label,data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#cleans CSV
def column2Float(dataset,column) :
	dataset[column]=[float(data.strip()) for data in dataset[column]]

#loads Data
def load_data(filename) :
	with open(filename, 'r') as f:
		data=csv.reader(f)
		rowList=[]
		for row in data:
			colList=[]
			for column in row :
				try :
					# fprint("column",column)
					colList.append(ast.literal_eval(column))
				except :
					print("csv error")
			rowList.append(colList)
		return rowList

#imports and cleans specified CSV
def importAndCleanCSV(filename) :
	loadedData=load_data(filename)

	# Apply to the loaded Swedish auto data here
	for i in range(len(loadedData)) :
		column2Float(loadedData,i)

	print("{0}:\ninstances: {1} \nfeaturs: {2}".format(filename, len(loadedData), len(loadedData[0])))
	return loadedData


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#takes in full instances, including the class, and returns euclidean distance
def get_euc_dist(inst1, inst2) :
	# print("inst1 {0}, inst2 {1}".format(inst1, inst2))
	return distance.euclidean(inst1, inst2[:-1])

def indev_euc_dist(inst1, inst2) :
	# print("inst1 {0}, inst2 {1}".format(inst1, inst2))
	return distance.euclidean(inst1, inst2)

def rssi_to_dist(rssi, n):
	# rssi=int(re.sub("[^0-9\-]", "", rssi))
	txPow=-50 #Standard transmission power for bluetooth, could be updated to be dynamic.
	return math.pow(10, ((txPow-rssi)/(n*10)))

def dist_to_n(rssi, dist):
	rssi=int(re.sub("[^0-9\-]", "", rssi))
	txPow=-50 #Standard transmission power for bluetooth, could be updated to be dynamic.
	return math.pow(10, ((txPow-rssi)/(n*10)))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


filename="3group.csv"
dataset=load_data(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
features=[]
log_features=[]
results=[]

# Loads in the specific format of the CSV

# for entry in dataset :
# 	device_loc=entry[1]
#
# 	for beacon in entry[0] :
#
# 		if beacon[1]>-100 : #this is to remove the outliers shown in figure 5
# 			dist=get_euc_dist(beacon[0],device_loc)
# 			results.append(dist)
# 			# fprint("beacon[1]",beacon[1])
# 			log_features.append([math.log10(abs(beacon[1]))])
# 			features.append([beacon[1]])

# lprint("coc",dataset)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This gets the center of the beacons for the instance
def get_center(beacon_list) :
	locations, rssi= zip(*beacon_list)
	min_x=locations[0][0]
	max_x=locations[0][0]
	min_y=locations[0][1]
	max_y=locations[0][1]
	for beacon in locations :
		if min_x>beacon[0] :
			min_x=beacon[0]
		if max_x<beacon[0] :
			max_x=beacon[0]
		if min_y>beacon[1] :
			min_y=beacon[1]
		if max_y<beacon[1] :
			max_y=beacon[1]
	return [(max_x-min_x)/2 + min_x, (max_y-min_y)/2 + min_y]

#conversts a list of locations to centered ones
def list_to_loc(values, center_list) :
	for i in range(len(values)) :
		values[i]=values[i]-center_list[i]

#centers an instance
def center_inst(instance) :
	center=get_center(instance[0])

	#center beacon vals
	for beacon in instance[0] :
		list_to_loc(beacon[0], center)

	#center relative rssi, even though this will be changed again
	list_to_loc(instance[1], center)

# #This converts all
# for instance in dataset :
# 	center_inst(instance)

# lprint("dataset",dataset)
'''All of the instances have been zero'd to their center '''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This section of code will "quadtrantize" the output data
#takes the zero'd location values and sorts them intro quadrants
def quatrandify(location) :
	quad=0
	if location[0]>0 :     #if x is on the right
		if location[1]>0 : #if y is on the top
			quad=1
		else :			   #if y is on the bottom
			quad=3
	else :				   #if x is on the left
		if location[1]>0 : #if y is on the top
			quad=0
		else :			   #if y is on the bottom
			quad=2
	return quad

# fprint("quadtrantized loc", dataset[0][1])
# fprint("quadtrantized loc", quatrandify(dataset[0][1]))
#
# for instance in dataset :
# 	instance[1]=quatrandify(instance[1])

# lprint("dataset",dataset)

'''
All of the instances have quadrants as outputs now

HOWEVER: I realized everything above this is practically useless
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This the proper way to handle the above.

features, results= zip(*dataset)

# lprint("feat",features)

#this generates the grid which will be used to assign values to locations. Values are assigned via grid_gen[x][y]
def grid_gen(side_length) :
	grid=[0]*side_length
	for row_index in range(side_length) :
		row=[0]*side_length
		for col_index in range(side_length) :
			row[col_index]=col_index*side_length+(row_index)
		grid[row_index]=row
	return grid
# fprint("grid", grid_gen(3))s


#gets the local minimum for x and y, to be used to prep for gridifying
def get_min_max(location_list) :
	locations, rssi= zip(*location_list)
	x,y=zip(*locations)
	return [[min(x), max(x)],[min(y),max(y)]]

def scale(min,max, val) :
	scaled_value=(val-min)/(max-min)
	# fprint("min",min)
	# fprint("max",max)
	# fprint("val",val)
	# fprint("out", scaled_value)
	return scaled_value

def rssi_to_dist(rssi, n):
	# rssi=int(re.sub("[^0-9\-]", "", rssi))
	txPow=-50 #Standard transmission power for bluetooth, could be updated to be dynamic.
	return math.pow(10, ((txPow-rssi)/(n*10)))

def standardize_rssi(min_rssi, max_rssi, rssi) :
	n=2.9#see https://www.slideshare.net/slideshow/embed_code/key/cJTaUJ7qXumc8V for chosen value, but it really shouldnt matter.
	min_val=rssi_to_dist(max_rssi, 1.6) #So here is where things get a little weird. I am going to use variable n values to account for in room attenuation.
	max_val=rssi_to_dist(min_rssi, 6)

	# fprint("min_rssi",min_rssi)
	# fprint("max_rssi",max_rssi)
	# min_val=min_rssi
	# max_val=max_rssi

	# print("min {0} max {1}".format(min_val,max_val))
	dist=rssi_to_dist(rssi,n)
	scaled_dist=scale(min_val,max_val,dist)
	# fprint("dist",dist)
	# fprint("scaled_dist",scaled_dist)
	return scaled_dist

#This, given a location and the corosponding data, returns grid location
def get_grid_loc(location, min_max_list,grid_side_length) :
	grid=grid_gen(grid_side_length)
	coords=location

	for i in range(len(coords)) :
		# print("({0})-({1})={2}".format(coords[i],min_list[i],coords[i]-min_list[i))
		min_val=min_max_list[i][0]
		max_val=min_max_list[i][1]

		if min_val!=max_val : #avoiding dividing by zero
			val=int(scale(min_val,max_val, coords[i])*grid_side_length)
		else : #the case where there is one row
			val=0

		if val>grid_side_length :
			coords[i]=grid_side_length-1
		elif val>=1 :	#because list indexing starts at one less, but -1 is not an index
			coords[i]=val-1
		else :
			coords[i]=0
	grid_loc=grid[int(coords[0])][int(coords[1])]
	return grid_loc

# fprint("loc",get_grid_loc([19, 4],get_local_min(dataset[0][0]),3))



#This converts whole instances to grid referenced
#So each location in a list equates to a square
def inst_to_grid(instance, grid_side_length,min_rssi, max_rssi) :
	beacon_locations=instance[0]
	actual_location=instance[1]
	locational_feature_list=[0]*grid_side_length**2
	local_min=get_min_max(beacon_locations)

	# fprint("local_min",beacon_locations)
	loc, rssi= zip(*beacon_locations)

	for i in range(len(loc)) :
		# print("RSSI")
		locational_feature_list[get_grid_loc(loc[i],local_min,grid_side_length)]=standardize_rssi(min_rssi, max_rssi, rssi[i])
	# actual_location=get_grid_loc(actual_location,local_min,grid_side_length)
	return[locational_feature_list,actual_location]



def get_grid_index(grid, val):
	for row in grid:
		if val in row:
			# fprint("grid", grid)
			x=grid.index(row)
			y=row.index(val)
			# print("val {0}, x {1}, y {2}".format(val, x, y))
			return [x, y]




def loc_to_pos(loc_list, grid_side_length) :
	grid=grid_gen(grid_side_length)
	to_return=[]
	for val in loc_list :
		indexes=get_grid_index(grid, val)
		# fprint("val",indexes)
		to_return.append(indexes)
	return to_return

def writeCSV(filename, csv_list) :
	#Write the data to a csv so I don't need to melt my computer every time I need this data
	with open(filename, 'w', newline='') as myfile:
		 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		 wr.writerows(csv_list)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRIDSIZE=4
location_list, out=zip(*dataset)

#Here I am getting the minimal and maximal rssi values across the entire dataset, which I will use to have a homogeniously scaled rssi output.
#HOWEVER There are some issues relating to outliers, which is causing my scaling to be way off.
flat_list=[item for group in location_list for item in group]
index, rssi = zip(*flat_list)
# lprint("location_list[0]",rssi)

# lprint("features", flat_list)
min_rssi=min(rssi)
max_rssi=max(rssi)
# fprint()


grid_data_set=[]
for intance in dataset :
	grid_data_set.append(inst_to_grid(intance,GRIDSIZE, min_rssi, max_rssi))

# fprint("mmmm", len(grid_data_set))
# lprint("gridified data", grid_data_set)
# writeCSV("3groupmultiregression.csv",grid_data_set)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#custom scoring
def cust_scorer(x_val, y_val) :
	# print("x={0}, y={1}".format(x_val, y_val))
	x_loc=loc_to_pos(x_val, 4)
	y_loc=loc_to_pos(y_val, 4)
	score=[]
	for i in range(len(x_val)) :
		# print("x={0}, y={1}".format(x_loc[i], y_loc[i]))
		if (x_loc[i]!=None and y_loc[i]!=None) : #So sometimes this function gets passed Nonetype, and I don't know what that signifies. Theres no neighbors?

			dist=indev_euc_dist(x_loc[i], y_loc[i])
			score.append(abs((dist+1)**2)) #This is to increase the weight of one
	return sum(score)


my_scorer = make_scorer(cust_scorer, greater_is_better=False)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''Its time to start machine learning'''
features, results=zip(*grid_data_set)
results_x,results_y=zip(*results) #This is where shit gets wack
#This converts the format because apparently some regressors can't handle floats?
lab_enc = preprocessing.LabelEncoder()
results_x = lab_enc.fit_transform(results_x)
results_y = lab_enc.fit_transform(results_y)





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#MODEL COMPARISON
models = []
if COMPARISON==True :
	models.append(('LR', LogisticRegression(solver='liblinear')))
	models.append(('LL', linear_model.LassoLars(alpha=.1)))
	models.append(('NB', GaussianNB()))
	models.append(('SVRR', SVR(kernel='rbf',gamma='auto')))
	models.append(('LIN', SVR(kernel='linear',gamma='auto')))
	models.append(('RI',linear_model.Ridge(alpha=.5)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_scores = []
y_scores=[]
regressors = []
scoring = 'neg_mean_squared_error'
# scoring = my_scorer
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, shuffle=True)
	fprint("model",model)
	cv_results_x = model_selection.cross_val_score(model, features, results_x, cv=kfold, scoring=scoring)
	cv_results_y = model_selection.cross_val_score(model, features, results_x, cv=kfold, scoring=scoring)
	x_scores.append(cv_results_x)
	y_scores.append(cv_results_y)
	regressors.append(name)
	msg = "X %s: %f (%f)" % (name, cv_results_x.mean(), cv_results_x.std())
	print(msg)
	msg = "Y %s: %f (%f)" % (name, cv_results_y.mean(), cv_results_y.std())
	print(msg)


# boxplot algorithm comparison
if COMPARISON==True :
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison, RMSE scoring metric, localized grid w/ seperated X-Y')
	ax = fig.add_subplot(121)
	plt.boxplot(y_scores)
	ax.set_xticklabels(regressors)
	ax.set_xlabel("Y Scores")
	ax = fig.add_subplot(122)
	plt.boxplot(x_scores)
	ax.set_xticklabels(regressors)
	ax.set_xlabel("X Scores")
	plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Chosen model:
model=linear_model.LassoLars(alpha=.04)
X_train, X_test, y_train, y_test = train_test_split(features, results, test_size=0.2)

# fprint("y_train",y_train)
y_train_x,y_train_y=zip(*y_train)
y_test_x,y_test_y=zip(*y_test)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Hyper Parementer tuning
param_grid = {'alpha':np.linspace(.015, .08, 30)}

def grid_optomize(model, param_grid, k_folds) :
	kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)
	grid_x = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1, scoring=scoring, verbose=1)
	grid_y = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1, scoring=scoring, verbose=1)

	grid_x.fit(X_train,y_train_x)
	grid_y.fit(X_train,y_train_y)

	# evaluate the best grid searched model on the testing data

	x_score= grid_x.score(X_test, y_test_x)
	y_score= grid_y.score(X_test, y_test_y)
	fprint("X score", x_score)
	fprint("Y score", y_score)
	print("[INFO] gridX search best parameters: {}".format(
		grid_x.best_params_))
	print("[INFO] gridY search best parameters: {}".format(
		grid_y.best_params_))





# grid_optomize(model, param_grid, 50)

# cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
# fprint("result", mean(cv_results))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
param_grid = {'alpha':np.logspace(-4, -0.5, 30)}
X score: -132.60533198550073
Y score: -81.43667842266635
[INFO] gridX search best parameters: {'alpha': 0.0196382800192977}
[INFO] gridY search best parameters: {'alpha': 0.05968456995122311}


param_grid = {'alpha':np.logspace(-2, -0.5, 30)}
X score: -132.873294032022
Y score: -63.26457666937824
[INFO] gridX search best parameters: {'alpha': 0.020433597178569417}
[INFO] gridY search best parameters: {'alpha': 0.0757373917589501}

param_grid = {'alpha':np.linspace(.015, .08, 30)}
X score: -137.99524652599632
Y score: -88.16141000337649
[INFO] gridX search best parameters: {'alpha': 0.028448275862068963}
[INFO] gridY search best parameters: {'alpha': 0.05982758620689655}
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#X_train, X_test, y_train_x, y_test_x = train_test_split(features, results_x, test_size=0.2)
#X_train, X_test, y_train_y, y_test_y = train_test_split(features, results_y, test_size=0.2)



print("Starting")
kfold = model_selection.KFold(n_splits=10, shuffle=True)
loc_pred_y = model_selection.cross_val_predict(model, X_train, y_train_y, cv=kfold, verbose=2, n_jobs=6)
loc_pred_x = model_selection.cross_val_predict(model, X_train, y_train_x, cv=kfold, verbose=2, n_jobs=6)
x_score = model_selection.cross_val_score(model,X_train, y_train_y, cv=kfold, scoring=scoring)
y_score = model_selection.cross_val_score(model,X_train, y_train_x, cv=kfold, scoring=scoring)

acc_score=(y_score+x_score)/2

# lprint("y_pred", y_pred)
fprint("loc_pred_x", len(loc_pred_x))
fprint("y_train_y", len(y_train_y))


y_test=list(zip(y_test_x, y_test_y))
y_pred=list(zip(loc_pred_x, loc_pred_y))

# lprint("y_pred", y_pred)

lprint("y_pred", y_train)
fprint("y_test", len(y_train))

dist_list=[]
for act, pred in zip(y_test, y_pred) :
	dist_list.append(indev_euc_dist(act, pred))
# lprint("dist_list",dist_list)


'''
need to qrite graphing function, plots predicted and actual values on a grid and draws a line between them
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This prints out lines between the real and predicted values
# for i in range(0, len(act_list), 2):
# 	plt.plot(*zip(pred_list[i],act_list[i]), 'k-',alpha=0.6, dashes=[6, 2])


# plot prediction and actual data
# plt.figure(0)
fig, ax = plt.subplots()
ax.plot(*zip(*y_train))
ax.plot(*zip(*y_pred), 'ro')
plt.title("Full room tuned classification")
ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

fig, ax2 = plt.subplots()
plt.title("Location difference histogram")
ax2.set_xlabel('Distance')
ax2.set_ylabel('# of differences @ distance')
ax2.hist(dist_list)


# fig, ax3 = plt.subplots()
# plt.title("Custom Scorer")
# ax3.set_ylabel('Custom Scorer Value')
# ax3.boxplot(my_score)

fig, ax4 = plt.subplots()
plt.title("Accuracy")
ax4.set_ylabel('accuracy')
ax4.boxplot(acc_score)



plt.show()
