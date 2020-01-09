#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
# # Xavier Quinn
# ## Assignment 6: Classification with a neuron
#
# ### Part 1: Perceptron classification
#
# The perceptron, as we saw in class, is the simpliest form of neural network consisting of a single neuron. Because it's so simple, it can only be used for two-class classification problems.
#
# The perceptron is inspired by a single neural cell, called a neuron. This accepts input signals via dendrites. Similarly, the perceptron receives inputs from examples of training data, that we weight and combine in a linear equation, called the activation function.
#
# activation = bias + sum(weight(i) * xi)
#
# You should notice the similarity between this, and the linear regression and logistic regression that we've implemented so far.
#
# Once we've computed the activation, we then transform it into the output value, using a transfer function (such as the step transfer function below)
#
# prediction = 1.0 IF activation >= 0.0, ELSE 0.0
#
# In order for this mechanism to work, we have to estimate the weights given in the activation function. Fortunately, we know how to do that using stochastic gradient descent.
#
# Each epoch, the weights are updated using the equation:
#
# w = w + learning_rate * (expected - predicted) * x
#
# Where you know that (expected - predicted) is the measure of error.
#
# This is enough information for you to implement the following (which will be closely related to previous assignments):
#
# - a predict function
#     - that takes a single instance, and a list of weights, where weights[0] is the bias
# - a stochastic gradient descent function
#     - that takes training data, learning rate and a number of epochs
#     - where the weights are first assigned zero scores, and then iteratively updated based on the formula
#         - w(i) = w(i) + learning_rate * (expected - predicted) * x(i)
#     - where you also update the bias based on the formula:
#         - bias = bias + learning_rate * (expected - predicted)
# - a perceptron function
#     - that takes training set, test set, learning rate and epochs
#     - that learns the weights using SGD
#     - then makes predictions over the test set using these weights
#     - and returns these predictions as a list
#
# I've given you a contrived data set for both your predict function, and for testing your SGD function. I've included sample output below.
#
# Then I want you to apply your classifier to the included sonar dataset, using the parameters given, as well as running a reasonable baseline comparison algorithm. You should perform a 3 fold cross validation. You can find out more about this data set here: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
#
# The extra twist here is that the data in the sonar data set should be converted to floats EXCEPT for the class (in the last position in each instance), that we should convert to an integer that represents...what? Currently, the class is a nominal category, and we should convert it to an integer: 1 for one class and 0 for the other. Also we will not normalize this data. Why not?

# In[ ]:


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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ##Methods:
# #General
#Format print
VERBOSE=False

def fprint(label, value) :
	try: VERBOSE
	except NameError: print("{0}: {1}\n".format(label, value))
	else:
		if VERBOSE :
			print("{0}: {1}\n".format(label, value))


#Quick Print
def qPrint(*args) :
	var=args[len(args)-1]
	for i in range(len(args)-1) :
		print(args[i], end ="")

	print("{0}: {1}".format(var, repr(eval(var)))) #yeah yeah, insecure methods, but I doubt anyone will be tryig to inject code into my ML hw.


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

def cross_validation_data(dataset, folds) :
	# fprint("dataset", dataset)
	# fprint("folds", folds)
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Evualuation Assistant functions
#normalizes a dataset
def normalize(dataset, minmax) :
	if (len(dataset[0])!=len(minmax)) :
		print("lengths do not match")
		print("{0} !={1}".format(len(dataset[0]),len(minmax)))
		exit()

	for data in dataset :
		for i in range(0,len(minmax)) :
			#fprint("i:{0}, data".format(i),data[i])
			data[i]= (data[i] - minmax[i][0] )/( minmax[i][1] - minmax[i][0])

#returns the min and max
def minmax(dataset) :
	cols=dataset[0]
	cols=[[val,val] for val in cols]
	for data in dataset :
		for i in range(0,len(cols)) :
			current=data[i]
			if current<cols[i][0] : #if current column data is less than stored, replace
				cols[i][0]=current
			if current>cols[i][1] :
				cols[i][1]=current
	return cols


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


def accuracy(actual, predicted) :
	counter=0
	fprint("acc actual", actual)
	fprint("acc pred   ", predicted)
	for i in range(0, len(actual)) :
		act=actual[i]
		pred=predicted[i]
		# fprint("pred   ", pred)
		# fprint("pred   ", round(pred))
		if (act==pred) :
			counter=counter+1;
		# fprint("count", counter)
	score=100.0*(counter/len(actual))
	fprint("score", counter)
	return score


def zeroRC(train, test) :
	yList=[inst[-1:][0] for inst in train]
	toReturn=[max(set(yList), key=yList.count) for i in test]
	fprint("toReturn", toReturn)
	return toReturn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Implement or copy your code here

# - a predict function
#     - that takes a single instance, and a list of weights, where weights[0] is the bias
def perceptronPredict(instance, weights) :
	activated=weights[0] #+ bias

	for i in range(1, len(instance)) :
		# fprint("Predict: Instance", instance[i-1])
		# fprint("Predict: weight  ", weights[i])
		mult=weights[i]*instance[i-1]
		# fprint("Predict: mult    ", mult)
		activated=activated+mult;

	predicted = 1.0 if activated >= 0.0 else 0.0
	return predicted


# - a stochastic gradient descent function
#     - that takes training data, learning rate and a number of epochs
#     - where the weights are first assigned zero scores, and then iteratively updated based on the formula
#         - w(i) = w(i) + learning_rate * (expected - predicted) * x(i)
#     - where you also update the bias based on the formula:
#         - bias = bias + learning_rate * (expected - predicted)
# returns coeffs
def SGD(train, learningRate, epochs) :
	weights=[0.0 for i in range(len(train[0]))]

	for i in range(0, epochs) :
		totError=0;

		for n in range(0, len(train)) :
			instance=train[n]
			expected=instance[-1:][0]
			predicted=perceptronPredict(instance,weights)
			# fprint("instance",instance)
			# fprint("expected",expected)
			# fprint("predicted",predicted)


			instError=expected-predicted
			totError+= instError**2
			# fprint("error",error)



			weights[0]=weights[0] + learningRate * instError #bias update

			for j in range(1,len(instance)) :
				weights[j]=weights[j] + learningRate * instError * instance[j-1]

		fprint("Epoch",i)
		fprint("Learning rate", learningRate)
		fprint("Total Error",totError)
		# fprint("coefficents", weights)
	return weights




# - a perceptron function
#     - that takes training set, test set, learning rate and epochs
#     - that learns the weights using SGD
#     - then makes predictions over the test set using these weights
#     - and returns just these predictions as a list
def perceptron(train, test, learningRate, epochs) :
	weights=SGD(train, learningRate, epochs)
	predictions=[]
	for testInst in test :
		predictions.append(perceptronPredict(testInst,weights))
	return predictions


#evaluate algorithm method
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

		fprint("Eval_Alg: actual", actual)
		fprint("Eval_Alg: *args", args)
		predicted = algorithm(trainData,test, *args)
		fprint("Eval_Alg: predicted", predicted)
		# fprint("Eval_Alg: trainData", trainData)

		result = metric(actual,predicted)
		# fprint("evaluation result", result)
		scores.append(result)

	return scores


# Contrived data
# Predict should work, given the weights below


dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

weights = [-0.1, 0.20653640140000007, -0.23418117710000003]


for row in dataset:
	prediction = perceptronPredict(row, weights)
	print("Expected={0}, Predicted={1}".format(row[-1], prediction))

# fprint("predict test", perceptronPredict(dataset[0], weights))

# Using your SGD function with a learning rate of 0.1, and 5 epochs, should give you:
#
#>epoch=0, lrate=0.100, error=2.000
#>epoch=1, lrate=0.100, error=1.000
#>epoch=2, lrate=0.100, error=0.000
#>epoch=3, lrate=0.100, error=0.000
#>epoch=4, lrate=0.100, error=0.000
#
#[-0.1, 0.20653640140000007, -0.23418117710000003]

SGD(dataset, .1, 5)


# Parameters for learning over real data

filename = 'sonar.all-data.csv'

loadedData=load_data(filename)


#I'm sure theres a better way to do this for this dataset, but this works.
loadedData=str(loadedData) #cast to string
loadedData=loadedData.replace('R', '0') #replace using strings built in method
loadedData=loadedData.replace('M', '1')
loadedData=ast.literal_eval(loadedData) #convert back to nested list


for i in range(len(loadedData)) :
	column2Float(loadedData,i)
dataset=loadedData
print("{0}:\ninstances: {1} \nfeaturs: {2}".format(filename, len(loadedData), len(loadedData[0])))


# dataset=importAndCleanCSV(filename)


folds = 3
learning_rate = 0.01
epochs = 500



print("SCORE")

# VERBOSE=True
perceptronScore = evaluate_algorithm(dataset, perceptron, folds, rmse_eval, learning_rate, epochs)
zeroRCScore = evaluate_algorithm(dataset, zeroRC, folds, rmse_eval)





VERBOSE=True
fprint('SGD_LOG', perceptronScore)
fprint('   best', min(perceptronScore))
fprint('  worst', max(perceptronScore))
fprint('   mean', stat.mean(perceptronScore))

fprint('zeroRC RMSE', zeroRCScore)
fprint('   best', min(zeroRCScore))
fprint('  worst', max(zeroRCScore))
fprint('   mean', stat.mean(zeroRCScore))


print("As we would have guessed, the perceptron greatly outpreforms ZERORC. I was briefly wondering about if I should use ZeroRC or ZeroRR, but I beleive it would not cause notable difference as the values are only 1 or 0")
