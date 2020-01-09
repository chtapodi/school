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
from scipy import stats

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
from sklearn.metrics import accuracy_score


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

from sklearn.metrics.scorer import make_scorer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ##Methods:
# #General
#Format print
# VERBOSE=False
VERBOSE=True

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

def writeCSV(filename, csv_list) :
	#Write the data to a csv so I don't need to melt my computer every time I need this data
	with open(filename, 'w', newline='') as myfile:
		 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		 wr.writerows(csv_list)
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



# lprint("feat",features)

#this generates the grid which will be used to assign values to locations. Values are assigned via grid_gen[x][y]
def cust_grid_gen(x_length, y_length) :
	grid=[0]*y_length
	for row_index in range(y_length) :
		row=[0]*x_length
		for col_index in range(x_length) :
			row[col_index]=col_index+x_length*(row_index)
		grid[row_index]=row
	return grid
# fprint("grid", grid_gen(3))s


#gets the local minimum for x and y, to be used to prep for gridifying
def get_min_max(location_list) :
	locations, rssi= zip(*location_list)
	x,y=zip(*locations)
	return [[min(x), max(x)],[min(y),max(y)]]

def scale(from_min,from_max,to_min, to_max, val) :
	# print("from_min {0},from_max {1},to_min {2}, to_max {3}, val {4}".format(from_min,from_max,to_min, to_max, val))
	from_range = (from_max - from_min)
	to_range = (to_max - to_min)
	return (((val - from_min) * to_range) / from_range) + to_min

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
def get_grid_loc(location, min_max_list,grid_x_length, grid_y_length) :
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
	actual_location=get_grid_loc(actual_location,local_min,grid_side_length)
	return[locational_feature_list,actual_location]



def get_grid_index(grid, val):
	for row in grid:
		if val in row:
			# fprint("grid", grid)
			y=grid.index(row)
			x=row.index(val)
			# print("val {0}, x {1}, y {2}".format(val, x, y))
			return [x, y]



def pos_to_coord(grid,pos) :
	for y in range(len(grid)) :
		for x in range(len(grid[y])) :
			# fprint("x",x)
			if grid[y][x]==pos :
				return [x,y]


def custom_loc_to_pos(loc_list, grid) :
	to_return=[]
	for val in loc_list :
		indexes=get_grid_index(grid, val)
		# fprint("val",indexes)
		to_return.append(indexes)
	return to_return

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#custom scoring
def cust_scorer(test_list, pred_list) :
	score=[]
	for test_val, pred_val in zip(test_list, pred_list) :
		test_coord=pos_to_coord(room_grid, test_val)
		# print("test_coord:{0} test_val:{1} pred_Val:{2}".format(test_coord, test_val, pred_val))
		pred_coord=pos_to_coord(room_grid, pred_val)
		if test_coord!=None and pred_coord!=None :
			dist=indev_euc_dist(test_coord, pred_coord)
			score.append((dist+1)**2) #This is to increase the weight of one

	return sum(score)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START REAL CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#making a scorer
my_scorer = make_scorer(cust_scorer, greater_is_better=False)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


filename="room_formated.csv"
dataset=load_data(filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRIDSIZE=4
fprint("DATASET", len(dataset[0]))



flat_dataset=[]
for entry in dataset :
	roomlist=entry[0]
	# fprint("roomlist",roomlist)
	full_room=[]
	for row in roomlist :
		for element in row :
			if element!=0 :
				# fprint("element", element)
				element=scale(-200,0, 0, 1, element)
				# fprint("element", element)
			full_room.append(element)
	flat_dataset.append([full_room,entry[1]])

# lprint("flat_dataset",flat_dataset)




features, results_loc=zip(*flat_dataset)

'''
So an interesting question is if it matters if the number of instances matches the number of classes, and I would think no, probably not.
'''

ROOMX=23
ROOMY=18
# ##Full Room
GRIDX=ROOMX
GRIDY=ROOMY

#1/5th room
# GRIDX=int(ROOMX/5)
# GRIDY=int(ROOMY/5)

#1/3th room
# GRIDX=int(ROOMX/3)
# GRIDY=int(ROOMY/3)


#generates grid in shape of room
room_grid=cust_grid_gen(GRIDX,GRIDY)
lprint("room_grid", room_grid)

results=[]
for coord in results_loc :
	# fprint("coord", coord)
	x=coord[1]
	y=coord[0]
	# print("x1:{0} y1:{1}".format(x, y))

	y=int(scale(0,ROOMY,0,GRIDY-1,y))
	x=int(scale(0,ROOMX,0,GRIDX-1,x))

	# print("x:{0} y:{1}".format(x, y))

	loc=int((room_grid[y][x]))
	results.append(loc)

# writeCSV("full_room5th_res.csv", zip(features, results))

'''
I need a way to shrink the room resulution, this is way too many classes.

'''

# lprint("results",results)
# fprint("TEST",pos_to_coord(room_grid,23))



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''Its time to start machine learning'''

def grid_optomize(model, param_grid, k_folds) :
	kfold = model_selection.KFold(n_splits=k_folds, shuffle=True)
	grid = GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, scoring=my_scorer, verbose=2)

	grid.fit(X_train,y_train)

	# evaluate the best grid searched model on the testing data

	acc = grid.score(X_test, y_test)
	fprint("score", acc)
	print("[INFO] grid search best parameters: {}".format(
		grid.best_params_))

''' MODEL SELECTION '''
model=SVC(kernel='rbf', gamma=35, C=242)   #3x4grid model
# model=SVC(kernel='rbf', gamma=3, C=1045) #6x7grid model
# model=SVC(kernel='rbf', gamma=.18, C=9650) #18x23grid model

#cy= −0.009042x^2+27.25x−83.76
#gy= 0.002635x^2−1.209x+49.13

#linearly predicted C=2525.37 for 9x11 grid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generate data
X_train, X_test, y_train, y_test = train_test_split(features, results, test_size=0.2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Hyper Parementer tuning
gamma_range = list(range(20,  75, 5))
C_range = list(range(10, 1000, 10))


# param_grid={'C': np.linspace(1040, 1060, 10)}




# param_grid={"C" :[10]}

#Comment out when not optomizing
# grid_optomize(model, param_grid, 50)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #CV tuning ish
# CV_range=np.linspace(10,50,5,dtype = int)
#
# accuracy_scores=[]
# euc_score_scores=[]
# fprint("range", len(CV_range))
#
# for i in CV_range :
# 	fprint("i",i)
# 	kfold = model_selection.KFold(n_splits=i, shuffle=True)
# 	acc=model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', verbose=2)
# 	ms=model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=my_scorer, verbose=2)
# 	lprint("acc", acc)
# 	lprint("ms",ms)
# 	fprint("acc", np.mean(acc))
# 	fprint("ms",np.mean(ms))
# 	accuracy_scores.append([i,np.mean(acc)])
# 	euc_score_scores.append([i,np.mean(ms)])
#
# #This should print the most accurate scores
# accuracy_scores.sort(key = lambda x: x[1])
# euc_score_scores.sort(key = lambda x: x[1])
#
# lprint("accuracy", accuracy_scores[:15])
# lprint("my_scorer", euc_score_scores[:15])
#
# plt.figure(0)
# plt.plot(accuracy_scores)



#
'''
RESULTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1/5TH ROOM
[INFO] grid search took 42.62 seconds
[INFO] grid search score: -438.5097
[INFO] grid search best parameters: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}


[INFO] grid search took 73.62 seconds
[INFO] grid search score: -438.5096679918783
[INFO] grid search best parameters: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}


[INFO] grid search took 3913.05 seconds
[INFO] grid search score: -438.5096679918783
[INFO] grid search best parameters: {'C': 205, 'gamma': 0.001, 'kernel': 'rbf'}


[INFO] grid search score: -438.5096679918783
[INFO] grid search best parameters: {'C': 130, 'gamma': 35.93813663804626, 'kernel': 'rbf'}

[INFO] grid search score: -438.5096679918783
[INFO] grid search best parameters: {'C': 240, 'gamma': 35, 'kernel': 'rbf'}


score: 0.8873239436619719

[INFO] grid search best parameters: {'C': 240}
0.8661971830985915


[INFO] grid search score: -438.5096679918783

[INFO] grid search best parameters: {'C': 245}
0.8661971830985915


So I have been getting the same score down to a scary level of precision over a wide range of tests, but it turns out that this is completely repeatable, and if I mess with the initial values enough, I can get different scores. This tells me that nothing I was optomizing mattered that much.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FULL room
param_grid={'kernel': ['rbf', "sigmoid"], 'gamma': np.logspace(-5,5,10),
					 'C': np.logspace(-5,5,10)}


score: -3463.995956507856

[INFO] grid search best parameters: {'C': 100000.0, 'gamma': 0.2782559402207126, 'kernel': 'rbf'}
score: 0.014084507042253521


[INFO] grid search best parameters: {'C': 10000.0, 'gamma': 0.31622776601683794, 'kernel': 'rbf'}
Starting

score: -3015.702193866757
[INFO] grid search best parameters: {'C': 9000, 'gamma': 0.1, 'kernel': 'rbf'}

score: -2836.8525893388214
[INFO] grid search best parameters: {'C': 9600, 'gamma': 0.2, 'kernel': 'rbf'}

score: -3094.188814698098
[INFO] grid search best parameters: {'C': 9600, 'gamma': 0.1875, 'kernel': 'rbf'}

score: -3322.5137803974626
[INFO] grid search best parameters: {'C': 9627.777777777777, 'gamma': 0.18333333333333332, 'kernel': 'rbf'}

score: -2649.327378080519
[INFO] grid search best parameters: {'C': 9650.0, 'gamma': 0.1875, 'kernel': 'rbf'}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1/3RD Room

score: -744.8366153280595
[INFO] grid search best parameters: {'C': 1000.0, 'gamma': 3.593813663804626, 'kernel': 'rbf'}


score: -685.7401153701779
[INFO] grid search best parameters: {'C': 1100.0, 'gamma': 3.0}

score: -727.7765920507645
[INFO] grid search best parameters: {'C': 1000.0}


score: -712.8954898311049
[INFO] grid search best parameters: {'C': 1050.0}


score: -627.8985428261926
[INFO] grid search best parameters: {'C': 1040.0}
'''




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
print("Starting")
kfold = model_selection.KFold(n_splits=100, shuffle=True)
y_pred = model_selection.cross_val_predict(model, X_train, y_train, cv=kfold, verbose=2, n_jobs=6)
my_score = model_selection.cross_val_score(model,X_train, y_train, cv=kfold, scoring=my_scorer)
acc_score = model_selection.cross_val_score(model,X_train, y_train, cv=kfold, scoring='accuracy')

# lprint("y_pred", y_pred)

dist_list=[]
for act, pred in zip(y_test, y_pred) :
	dist_list.append(indev_euc_dist(act, pred))
# lprint("dist_list",dist_list)



'''
I should graph with low resolution scale and full res as well. Plot beacon locations for full res.
'''


pred_list=custom_loc_to_pos(y_pred,room_grid)
act_list=custom_loc_to_pos(y_test,room_grid)
# lprint("ypred",act_list)
# plt.scatter(*zip(*act_list), alpha=.1)
# plt.scatter(*zip(*pred_list), alpha=.1)




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
ax.scatter(y_train, y_pred, edgecolors=(0, 0, 0))
plt.title("full Resolution Tuned Classification")
ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

fig, ax2 = plt.subplots()
plt.title("full Resolution Location Difference Histogram")
ax2.set_xlabel('Distance')
ax2.set_ylabel('# of differences @ distance')
ax2.hist(dist_list)


fig, ax3 = plt.subplots()
plt.title("full Resolution Custom Scorer")
ax3.set_ylabel('Custom Scorer Value')
ax3.boxplot(my_score)

fig, ax4 = plt.subplots()
plt.title("full Resolution Accuracy")
ax4.set_ylabel('accuracy')
ax4.boxplot(acc_score)



plt.show()


'''
I realized that the method I implemented in previous research will not work with machine learning, as it would be really dumb.

So I beleive my current plan is to arrange and feed in the relative beacon locations and the rssi value as features, and have the output be a classifier of regions.
If I am going this route, it may make sense to degrade the beacon locations in the same manor.

I have decided to set the reference point as the average location for all beacons, and then divide this into four quadrants

So now the problem is that it is not feasible to have "grouped" features, such as x,y,r

The solution I am going to apply to this is taking the groups of 3 instances, and assigning them coordinates with a gridspace of 16, and then putting their rssi data at the relative location of the beacon, which of course means that I should not be using quadrants for the data


This is now complete, the time to classify is here.
Need multiclass(?), I am pretty sure


## TODO:
See if its possible to make my own scoring accuracy thing based on euc dist, as the grid thing is not good due to the other thing.
This is Done

I was able to get a solid 86% accuracy on a really small room, so now I am trying for the full resolution room.



'''
