#!/usr/bin/env python
# coding: utf-8


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Author: Xavier Theo Quinn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I don't really know how jupityer parses things, so hopefully this does not mess you up

#Imports
import csv
import matplotlib.pyplot as plt
import math
import copy
import random
import statistics as stat
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ##Methods:
# #General
#Easy print
def fprint(label, value) :
	print("{0}: {1}\n".format(label, value))

# #CSV functions

#cleans CSV
def column2Float(dataset,column) :
	dataset[column]=[float(data) for data in dataset[column]]

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #algorithms and related

#predict
#y = b0 + b1*x1 + b2*x2 + ... + bN*xN
def predict(instance, coefficients) :
	yVal=coefficients[0];
	# fprint("INSTANCEP", instance)
	for i in range(1, len(instance)) :
		# fprint("MULT is {0}*".format(coefficients[i]), instance[i-1])
		mult=coefficients[i]*instance[i-1]
		yVal=yVal+mult;
	# fprint("yVal", yVal)
	return yVal;

# coefficientsSGD
# generates coefficents for stochastic gradient descent
def coefficientsSGD(train, learning_rate, epochs) :
	coefficents=[0.0 for i in range(len(train[0]))]
	# fprint("TRAIN", train)
	for i in range(0, epochs) :
		error=0;
		for n in range(0, len(train)) :
			# fprint("INSTANCEC",train[n])
			predicted=predict(train[n],coefficents)
			instErr=predicted-train[n][-1:][0]

			error += math.sqrt(instErr**2)
			coefficents[0]=coefficents[0] - learning_rate * instErr
			for j in range(1,len(train[n])) :
				coefficents[j]=coefficents[j] - learning_rate * instErr * train[n][j-1]

		# fprint("Epoch",i)
		# fprint("Learning rate", learning_rate)
		# fprint("Total Error", error)
		# fprint("coefficents", coefficents)
	return coefficents

#ZeroRR
def zeroRR(train,test) :
	fprint("ZERORR: train", train)
	fprint("ZERORR: test", test)


	Y=0
	for item in train :
		Y+=item[-1:][0]
	meanVal=Y/len(test)
	mean=train
	toReturn=[[item[0], meanVal] for item in test]
	fprint("ZERORR: toReturn", toReturn)
	return toReturn

#MLR
def mlr(train,test,learning_rate,epochs) :
	fprint("train", train)
	fprint("test", test)
	coefs = coefficientsSGD(train, learning_rate, epochs)
	predictions=[]
	for i in range(len(test)) :

		predictedVal=predict(test[i],coefs)
		predictions.append(predictedVal)

	# fprint("predicted", predictions)
	return predictions

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
	fprint("rmse actual", actual)
	fprint("rmse predicted", predicted)
	result=0
	for i in range(len(actual)) :
		fprint("rmse actual", actual[i])
		fprint("2 predicted", predicted[i])
		result=result+(predicted[i]-actual[i])**2
	# result=[(pred-act)**2 for act, pred in zip(actual, predicted)]

	average=result/len(actual)
	return average

# end copy block
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # CSC-321: Data Mining and Machine Learning
# ## Assignment 5: Classification
#
# ### Part 1: Cross-validation for training and testing
#
# So far we've used the same data for training and testing. That's definitely NOT a great idea. We'll now implement k-fold cross-validation. We'll pass k as a parameter, so we can decide how many folds we want to make. The general idea is that we need to split the data into subsections for training and for testing. Refer to your notes to remind yourself how cross-validation works.
#
# (a) Create a function called cross_validation_data(dataset, folds). This function is going to split our data in k folds, where k is the parameter given. The function should create (and ultimately return) a new list. We need to take a shallow copy of the data set, and operate on it. We need to determine how much data will be in each fold, by taking the length of our dataset, and dividing it by the number of folds, probably using integer division.
#
# Then we need to loop number of folds times, creating a new fold and populating that fold with data from our copy of the dataset. Roughly speaking that means:
#
# - while the amount of data in the current fold is less than the number we determined above for how much data SHOULD be in each fold
#     - choose a random instance from our copy of the data set
#         - HINT: You can use functions from the random library to help you (https://docs.python.org/3/library/random.html)
#     - Place the chosen instance into our current fold
#     - REMOVE the instance from the copy of the data
#
#     - append the new fold to our list for returning
# - Continue until we've populated all the folds
#
# Before using whatever random method you choose to select values, we'll set the seed to 1. For this assignment it will mean that your results should be reproducably the same every time you run it, which can help with the testing. I've done that for you below.
#
# As an example, if we have a data set with 8 instances, and we split it into 4 folds, we'll have four sublists, each with two instances. E.g.
# if our input dataset was [[1,1], [2,1], [4,2], [6,1], [7,3], [8,4], [9,6], [5,2]]
#
# our output COULD be:
#
# [[[2,1], [4,2]], [[5,2], [8,4]], [[6,1],[9,6]], [[7,3], [1,1]]]
#
# NOTE the additional level of nesting, with respect to what happens in the function we write next. Create yourself a contrived test set, and try the function out. REMEMBER, that an instance can only appear in a single fold.
#
# This is a naive way to do this, and the resulting folds will definitely not be stratified, but it's better than anything we've done so far.
#

# In[ ]:



random.seed(1)


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


testSet=[[1,1], [2,1], [4,2], [6,1], [7,3], [8,4], [9,6], [5,2]]
testSet= cross_validation_data(testSet, 4)


# (b) Now you need to amend the evaluate algorithm function I've given you previously. I've changed the signature to include a folds parameter. This function needs to:
#
# - Use your function from (a) above, to create k folds of data
# - Create an empty list of scores
# - For each fold in your evaluation
#     - (1) set up the training set for that fold
#     - (2) You can think of that as REMOVING the fold which will be used for evaluation in this fold from the data you returned from (a)
#     - (3) BE CAREFUL to always operate on copies of the data - you don't want to mess up your original splits
#     - (4) You then need to 'flatten' the remaining data in the current training set
#         - (5) For example if the data was as given above:
#             - (6) [[[2,1], [4,2]], [[5,2], [8,4]], [[6,1],[9,6]], [[7,3], [1,1]]]
#         - (7) We would remove ONE fold first for testing (let's say the last one, but it will be each fold in turn)
#         - (8) Leaving [[[2,1], [4,2]], [[5,2], [8,4]], [[6,1],[9,6]]] for training
#         - (9) We need to convert that into: [[2,1], [4,2], [5,2], [8,4], [6,1],[9,6]]
#         - (10) This is the usual format we pass data into our algorithms
#     - (11) We also need to prepare our test set, which is the held-out fold from step (2) above
#     - (12) Append the instances from this held-out fold to a new list making sure the last value of each instance is NONE
#     - Once this is done, you should have a training set, comprised on k-1 folds of instances, and a test set, comprised of 1 fold of instances. We're now ready to evaluate our algorithm just as we have previously, using a training set, a test set and a specified evaluation metric. This should return a score to us. Instead of using that score directly, we should add it to our score list
# - At the end of this function, RETURN the complete list of scores. Therefore, if we did a 5 fold cross validation, we should get a list of 5 scores.
#
# I've given you the (very) bare bones of the function below.

# In[ ]:



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
		fprint("eval actual", actual)

		# fprint("trainData", trainData)

		predicted = algorithm(trainData,test, *args)
		fprint("Eval_ALg: predicted", predicted)

		fprint("Eval_ALg: trainData", trainData)


		# fprint("actual", actual)

		result = metric(actual,predicted)
		# fprint("evaluation result", result)
		scores.append(result)

	return scores


#----------------------------------------------------------------




# ### Part 2: Applying cross-validation to real data
#
# To test the function you wrote above, let's apply it to the multivariate linear regression you wrote last week. Copy the function you wrote above to the cell below, along with all the code you need for BOTH MLR and zeroRR to work.
# Use the same parameters I gave you last week (a learning rate of 0.01 and 50 epochs of training), run MLR using a cross-validation of 5 folds. PRINT out the list of RMSE scores obtained on each fold. Then run zeroRR.
# Also print the LOWEST score obtained (that's the best), the highest score (that's the worst) and the mean RMSE score. See for yourself the variance in scores you can obtain using a cross-validation approach.

# In[ ]:


dataset = importAndCleanCSV("winequality-white.csv")
# fprint("cleaned and folded dataset", dataset)
normalize(dataset, minmax(dataset))
# fprint("normalized dataset        ", dataset)

learning_rate = 0.01
epochs = 5
print("MLR START")
mlr_result = evaluate_algorithm(dataset,mlr,5,rmse_eval,learning_rate,epochs)
print("ZERORR")
zeroRR_result = evaluate_algorithm(dataset,zeroRR, 5,rmse_eval)

fprint('MLR RMSE', mlr_result)
fprint('    best', min(mlr_result))
fprint('   worst', max(mlr_result))
fprint('    mean', stat.mean(mlr_result))
fprint('zeroRR RMSE', zeroRR_result)





# ### Part 3: Classification with Logistic Regression
#
# Everything so far has been a regression task - predicting a numeric value. We've moved on to talk about classification in class, so let's implement our first basic classifier. This is the same idea as linear regression, but we're going to predict one of two binary classes, using logistic regression.
#
# The general outline for logistic regression is the same as for multivariate linear regression. We're going to need a function to make predictions, and a function to learn coefficients.
#
# (a) The formula for making a prediction, predY, for logistic regression is:
#
# predY = 1.0 / 1.0 + e^-(b0 + b1 * x1 + ... + bN * xN)
#
# Where b0 is the intercept or bias, bN is the coefficient for the input variable xN, and e is the base of the natural logarithms, or Euler's number. We can use the python math library which has an implementation of e called math.exp(x): https://docs.python.org/3/library/math.html
#
# The formula given above is an implementation of a sigmoid function (a commonly used, s-shaped function that can take any input value and produce a number between 0 and 1).
#
# We will assume there can be multiple input features (x1, x2 etc) not just a single value, and that each input feature will have a corresponding coefficient (b1, b2 etc).
#
# Write your predict function, that will take an instance, and a list of coefficients, and return a prediction. In the list of coefficients, assume coefficient[0] corresponds to b0. This will be very similar to your predict function from last week.
#

# In[ ]:


# Write your predict function here

def logistic_predict(instance, coefficients) :
	expY=coefficients[0];
	for i in range(1, len(instance)) :
		mult=coefficients[i]*instance[i-1]
		expY=expY+mult;

	return 1.0/(1.0+math.exp(-expY))



# We can test your predict function on the contrived dataset below. It includes TWO input variables and a class (Y) for each instance. The class is either 0 or 1.
#
# (b) Graph this data, x1 against x2, using different colored points for the two classes. Include a legend, showing which color corresponds to which class.
#
# (c) Call your predict function on each instance in the contrived data set, using the coefficient list given below. Get the predicted class from your function, and print (for each instance), the expected class, the predicted value AND the predicted class. In order to get the predicted class from the value predicted, we need to do rounding. There is a round() function that can help you. If it works correctly, you should predict the correct class of each instance in the dataset.

# In[ ]:


# Here's the contrived data set

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


# Do the graphing here

plt.title('Logistic Prediction')
plt.plot(dataset[0][0], dataset[0][2], 'b^', label='x1')
plt.plot(dataset[0][1], dataset[0][2], 'g^', label='x2')
#god I'm so sorry, this is such shitty coding practice, but I have a surprise assignment due at the same time.
for i in range(1, len(dataset)-1) :
	plt.plot(dataset[i][0], dataset[i][2], 'b^')
	plt.plot(dataset[i][1], dataset[i][2], 'g^')
plt.legend()
# plt.show()


# Call your predict function on the data here, using the following coefficients

coeffs = [-0.406605464, 0.852573316, -1.104746259]

for instance in dataset :
	#the expected class, the predicted value AND the predicted class.
	result=logistic_predict(instance, coeffs)
	fprint("\n Expected", instance[-1:][0])
	fprint("    value", result)
	fprint("predicted", round(result))



# (d) Above I gave you coefficients. Just as with MLR, we need to estimate the coefficients for a data set. To do that, we're going to use stochastic gradient descent. The algorithm is exactly the same as for multivariate linear regression except for the following two things.
#
# b0 is computed by:
#
# b0 = b0 + learning_rate * error * predictedY * (1.0 - predictedY)
#
# and bN is computed by:
#
# bN = bN + learning_rate * error * predictedY * (1.0 - predictedY) * xN
#
# for all coefficients b1..bN
#
# Remember, to calculate the error, we run the algorithm with default coefficients and perform prediction, then get the error by subtracting the predictedY from the actual Y value.
#
# Refer back to Assignment 4 for the complete algorithm
#
# (d) Apply your function to the contrived dataset given above, using the learning rate of 0.3, and 100 epochs. Print the resulting coefficients. I've shown my last 5 epochs of this example.
#

# In[ ]:


# Write your function sgd_log(dataset, learning_rate, epochs) here

def sgd_log(dataset, learning_rate, epochs) :
    coefficents=[0.0 for i in range(len(dataset[0]))]
    for i in range(0, epochs) :
        error=0;
        for n in range(0, len(dataset)) :
            instance=dataset[n]
            predictedY=logistic_predict(instance,coefficents)
            instErr=instance[-1:][0]-predictedY

            error += math.sqrt(instErr**2)
            coefficents[0]=coefficents[0] + learning_rate * instErr * predictedY * (1.0 - predictedY) #b0 = b0 + learning_rate * error * predictedY * (1.0 - predictedY)
            for j in range(1,len(instance)) :

                coefficents[j]=coefficents[j] + learning_rate * instErr * (1.0 - predictedY) * instance[j-1] #bN = bN + learning_rate * error * predictedY * (1.0 - predictedY) * xN

        fprint("Epoch",i)
        fprint("Learning rate", learning_rate)
        fprint("Total Error", error)
        fprint("coefficents", coefficents)
    return coefficents


# Call your function using the parameters given here.

coeffs=sgd_log(dataset, .3, 10)

for instance in dataset :
	#the expected class, the predicted value AND the predicted class.
	result=logistic_predict(instance, coeffs)
	fprint("\n Expected", instance[-1:][0])
	fprint("    value", result)
	fprint("predicted", round(result))

# Example output
#
#>epoch=95, lrate=0.300, error=0.023
#>epoch=96, lrate=0.300, error=0.023
#>epoch=97, lrate=0.300, error=0.023
#>epoch=98, lrate=0.300, error=0.023
#>epoch=99, lrate=0.300, error=0.022


# ### Part 4: Applying classification to real data
#
# In this final section, we'll do the following things.
#
# (a) We need a function for calculating accuracy. It will take a list of actual class values, and predicted class values. If the actual value of an instance and the predicted value of an instance are the same, increment a counter. In the end, return the value of this counter divided by the length of the actual values list, multiplied by 100 - so we are returning a percentage of the classification we got correct. This function should be called accuracy(actual, predicted).
#
# (b) We need a baseline function. Create a function called zeroRC(train, test). This function should take in the training data, and find the most common value of Y in the training data. It should then return a list of predictions the same length as the test data, containing ONLY this value that was most common in the training data.
#
# (c) I've given you the diabetes data set. You can find more about this data set here: https://www.kaggle.com/uciml/pima-indians-diabetes-database
#
# You are going to:
#
# - load the data
# - print out some basic information about the data (number of instances, features)
# - convert each string value to float (for all columns)
# - normalize all columns in the range 0-1
# - perform a 5-fold cross validation
#     - using logistic regression
#     - using a learning rate of 0.1, and 100 epochs
# - collect predicted scores
# - print the min, max and mean scores
# - repeat the above, using zeroRC as the algorithm
# - offer me some write up of the results
#
#
#

# In[ ]:


# Do all the code here
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


def logistic_regression(train,test,learning_rate,epochs) :

	coefs = sgd_log(train, learning_rate, epochs)
	predictions=[]
	for i in range(len(test)) :
		# fprint("testI", test[i])
		predictedVal=predict(test[i],coefs)
		predictions.append(predictedVal)

	fprint("logistic predicted", predictions)
	return predictions

diabset = importAndCleanCSV('pima-indians-diabetes.csv')
# fprint("cleaned and folded dataset", diabset)
normalize(diabset, minmax(diabset))
# fprint("normalized diabset        ", diabset)


learning_rate = 0.1
epochs = 20
folds=5
print("LOGISTIC REGRESSION START")
logistic_regression_result = evaluate_algorithm(diabset,logistic_regression,folds,accuracy,learning_rate,epochs)
print("ZERORC START")
zeroRC_result = evaluate_algorithm(diabset,zeroRC, folds,accuracy)

fprint('SGD_LOG', logistic_regression_result)
fprint('   best', min(logistic_regression_result))
fprint('  worst', max(logistic_regression_result))
fprint('   mean', stat.mean(logistic_regression_result))

fprint('zeroRC RMSE', zeroRC_result)
fprint('   best', min(zeroRC_result))
fprint('  worst', max(zeroRC_result))
fprint('   mean', stat.mean(zeroRC_result))



# Write up your observations on the experiment here
#I got it to what I thought was a good place before work, apparently was not, very dissapointing, out of time.
#
