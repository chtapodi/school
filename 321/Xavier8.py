#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
# # Your Name: Xavier Quinn
# ## Assignment 8: K-Nearest Neighbor
#
# ### Part 1: Implementation
#
# For this assignment, I'm going to let you break down the implementation as you see fit. You're going to implement KNN. In brief, this means:
#
# - calculating euclidean distance between a test instance, and a training instance
# - iterating through all the training instances, and storing distances in a list
# - returning the k nearest neighbors
# - making a prediction by choosing the class that appears the most in the k nearest neighbors
#
# To calculate euclidean distance between an instance x1 and an instance x2, we need to iterate through the features (excluding the class) of the two instances (for i features) and for each take the difference of (x1[i]) - (x2[i]), squaring that difference, and summing over all features. At the end, take the square root of the total. In other words:
#
# $$distance=\sqrt{\sum_{i=1}^n (x1_{i} - x2_{i})^2}$$
#
# I didn't really need to include this equation, but now you have an example of embedding a latex equation inside markdown. You're welcome.
#
# I would strongly suggest you follow the implementation outline of previous algorithms in terms of the functions you use, but I'm leaving it up to you.
#
# Below is the same contrived dataset you've used before. If your code works, you should be able to take an instance of this data, and compare it to all the others (including itself, where the distance SHOULD be 0). You should be able to select the k-nearest neighbors, and make a prediction based on the most frequently occuring class.
#
# Make sure you create a knn function that takes a training set, a test set and a value for k.
#

# In[1]:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Author: Xavier Theo Quinn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Imports
import csv
import matplotlib.pyplot as plt
import math
import copy
import random
import statistics as stat
import ast
from scipy.spatial import distance
from collections import Counter

from string import ascii_lowercase
from datetime import datetime
from random import randint
import itertools


from statistics import mean

import numpy as np
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ##Methods:
# #General
#Format print
VERBOSE=False
# VERBOSE=True

def fprint(label, value) :
	try: VERBOSE
	except NameError: print("{0}: {1}\n".format(label, value))
	else:
		if VERBOSE :
			print("{0}: {1}\n".format(label, value))


def lprint(label,toPrint) :
	print(label)
	for data in toPrint :
		# for variableName in data :
		fprint("entry",data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#from prevuous assignments

# #CSV functions

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
					colList.append(column)
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

#cross validates a data
def cross_validation_data(dataset, folds) :
	foldSize=(int)(len(dataset)/folds)
	toFold=copy.deepcopy(dataset)
	folded=[]
	for foldIndex in range(folds) :
		sublist=[]
		for instIndex in range(foldSize) :
			#generate nested list
			sublist.append(toFold.pop(random.randint(0,len(toFold)-1)))
		#put sublist into folded list
		folded.append(sublist)

	return folded



#rmse_eval
def rmse_eval(actual,predicted) :
	# fprint("actual list", actual)
	# fprint("predicted list", predicted)
	result=0
	for i in range(len(actual)) :
		# fprint("RMSE: actual", actual[i])
		# fprint("RMSE: predicted", predicted[i])
		result=result+(predicted[i]-actual[i])**2

	average=result/len(actual)
	return average

#ZeroRC
def zeroRC(train, test) :
	yList=[inst[-1:][0] for inst in train]
	toReturn=[max(set(yList), key=yList.count) for i in test]
	fprint("toReturn", toReturn)
	return toReturn

#ZeroRR
def zeroRR(train,test) :
	# mean(yTrain[-1)
	meanVal=list(map(mean, zip(*train)))[-1]
	return [meanVal for item in test]

#accuracy measure
def accuracy(actual, predicted) :
	counter=0
	for i in range(0, len(actual)) :
		act=actual[i]
		pred=predicted[i]
		if (act==pred) :
			counter=counter+1;
	score=100.0*(counter/len(actual))
	return score

def evaluate_algorithm(dataset, algorithm, folds, metric, *args):
	scores = []
	folded= cross_validation_data(dataset, folds)

	for i in range(len(folded)) :
		trainData=copy.deepcopy(folded) #these two lines seperate the fold at the index from the rest of the folds
		test=trainData.pop(i)

		#the Flattening
		tmpData=[]
		for fold in trainData :
			for instance in fold :
				tmpData.append(instance)
		trainData=copy.deepcopy(tmpData) #I can't think of a reason for why this should be a deep copy, but might as well

		actual=[]
		#Prepares test Set
		for inst in test :
			actual.append(inst[-1:][0])
			inst[len(inst)-1]=None


		predicted = algorithm(trainData,test, *args)
		result = metric(actual,predicted)
		scores.append(result)

		fprint("Eval_Alg: actual", actual)
		fprint("Eval_Alg: predicted", predicted)
		fprint("Eval_Alg: trainData", trainData)
		fprint("evaluation result", result)

	return scores



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Contrived data set

dataset = [[3.393533211,2.331273381,1],
	[3.110073483,1.781539638,1],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,1],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]


#takes in full instances, including the class, and returns euclidean distance
def get_euc_dist(inst1, inst2) :
	#hopefully this is okay, I was already using it for my final project
	return distance.euclidean(inst1[:-1], inst2[:-1])


#sorts by euclidean distance
def sort_by_dist(to_comp, training_set) :
	to_sort=[]
	for inst in training_set :
		to_sort.append([get_euc_dist(to_comp,inst),inst])
	to_sort.sort(key=lambda x: x[0])
	return to_sort

#gets a list of k nearest neighbors
def get_k_neighbors(test_inst, train_set, k) :
	sorted_set=sort_by_dist(test_inst, train_set)
	nearest=[]
	for i in range(k) :
		nearest.append(sorted_set[i][1])
	return nearest

#classification predict
def knn_class_predict(class_list) :
	c = Counter(class_list)
	k_most_freq=c.most_common(1)
	return k_most_freq[0][0]

#Regression Predict
def knn_reg_predict(class_list) :
	return stat.mean(class_list)


#KNN classifier
def knn_class(train_set, test_set, k):
	predicted_set=[]
	for test_inst in test_set :
		Neighbor_list=get_k_neighbors(test_inst, train_set, k)
		class_list=[x[-1] for x in Neighbor_list]
		predicted_set.append(knn_class_predict(class_list))
	return predicted_set
#KNN regression
def knn_reg(train_set, test_set, k):
	predicted_set=[]
	for test_inst in test_set :
		Neighbor_list=get_k_neighbors(test_inst, train_set, k)
		class_list=[x[-1] for x in Neighbor_list]
		predicted_set.append(knn_reg_predict(class_list))
	return predicted_set

# knn(dataset, testdataset,3)

# lprint("sd",knn(dataset, testdataset,3))

# ### Part 2: Working with real data
#
# Apply the KNN algorithm above to the abalone data set. You can find more about it here: http://archive.ics.uci.edu/ml/datasets/Abalone
#
# You will need to make the class value into an integer class. Run a 5-fold cross-validation, with k set as 5. Also run a classification baseline. Report on classification accuracy, and write up some results.
#

# In[ ]:
filename="abalone.csv"
loadedData=load_data(filename)
loadedData=str(loadedData) #cast to string
loadedData=loadedData.replace('M', '0') #replace using strings built in method
loadedData=loadedData.replace('F', '1')
loadedData=loadedData.replace('I', '2')
loadedData=ast.literal_eval(loadedData) #convert back to nested list

#convert all values to float (You said make the last int, but I don't see why float won't work. Hopefully I'm not overlooking something)
for i in range(len(loadedData)) :
	loadedData[i].append(loadedData[i].pop(0))
	column2Float(loadedData,i)
dataset=loadedData

# lprint("",dataset)


folds = 5
k=5
#						dataset, algorithm, folds, metric, *args
knn_class_score=evaluate_algorithm(dataset, knn_class, folds, accuracy, k)
zeroRCScore = evaluate_algorithm(dataset, zeroRC, folds, accuracy)
#
#





# #### Write up results here
#
'''
knn Classification: [49.82035928143713, 54.97005988023952, 53.89221556886228, 56.40718562874252, 47.4251497005988]
   best: 56.40718562874252
  worst: 47.4251497005988
   mean: 52.50299401197605

zeroRC Classification: [35.688622754491014, 34.49101796407186, 37.24550898203593, 36.40718562874252, 39.04191616766467]
   best: 39.04191616766467
  worst: 34.49101796407186
   mean: 36.5748502994012

Right off the bat, its aparent that KNN grossly outperformed ZeroRC. The worst KNN accuracy score is better than the best ZeroRC accuracy score by such a margine that the difference between the best and worst KNN scores is approximately the same as the difference between the worst KNN score and the best ZeroRC score, which is a significant indicator of how much beter it did.
However, at best, the KNN classifier is only 56% accurate. As this dataset has 3 classes, this value is significantly better than a 3-sided coin flip, but as a direct percentage, it is not something I would feel comfortable relying on.

'''
# ### Part 3: KNN regression
#
# We can also run KNN as a regression algorithm. In this case, instead of predicting the most common class in the k nearest neighbors, we can assign a predicted value that is the mean of the values in the k neighbors.
#
# Make this change to your algorithm (presumably by simply implementing a new predict function below, because you divided your code up sensibly in Part 1), and run the abalone data as a regression problem (convert the class data to a float, before running the algorithm). Use the same number of folds and the same k value as before. Also run a regression baseline and report RMSE values for both. Give me some explanation of the results, both standalone and in comparison to the classification results above.
#

# In[ ]:

knn_reg_score=evaluate_algorithm(dataset, knn_reg, folds, rmse_eval, k)
zerorr_score=evaluate_algorithm(dataset, zeroRR, folds, rmse_eval)


VERBOSE=True
fprint('knn Classification', knn_class_score)
fprint('   best', max(knn_class_score))
fprint('  worst', min(knn_class_score))
fprint('   mean', stat.mean(knn_class_score))
#
fprint('zeroRC Classification', zeroRCScore)
fprint('   best', max(zeroRCScore))
fprint('  worst', min(zeroRCScore))
fprint('   mean', stat.mean(zeroRCScore))

fprint('knn regression', knn_reg_score)
fprint('   best', min(knn_reg_score))
fprint('  worst', max(knn_reg_score))
fprint('   mean', stat.mean(knn_reg_score))
#
fprint('zeroRR regression', zerorr_score)
fprint('   best', min(zerorr_score))
fprint('  worst', max(zerorr_score))
fprint('   mean', stat.mean(zerorr_score))


# #### Write up results here
#
#

'''
knn Regression: [0.49264670658682963, 0.571544910179644, 0.5258922155688655, 0.5913772455089853, 0.5508023952095841]
   best: 0.49264670658682963
  worst: 0.5913772455089853
   mean: 0.5464526946107817

zeroRR Regression: [0.6718664168668643, 0.6881516906307125, 0.6901448599806378, 0.6988192477320826, 0.6788411201549069]
   best: 0.6718664168668643
  worst: 0.6988192477320826
   mean: 0.6855646670730409


 Once again KNN has outperformed the baseline, and once again the worst KNN is better than the best baseline value.
 As the range of values is between 0-2, these scores are not particularily low, but KNN is not best suited for regression.

 I was initially pondering about using error feedback to change the K value, but after thinking about how KNN works, I realized that that makes very little sense, as k=1 will always have a perfect score for training, but is a great example of overfitting.
'''
