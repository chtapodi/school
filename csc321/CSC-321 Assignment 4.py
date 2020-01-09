
#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
# ## Assignment 4: Multivariate Linear Regression
#
#
# For more complex learning, we cannot simply read our coefficients (b0 and b1 for SLR) from the data. We have to approximate them because there are many more of them. If you remember, the function for SLR was:
#
# y = b0 + b1*x
#
# For the more complicated Multivariate Linear Regression the function is:
#
# y = b0 + b1*x1 + b2*x2 + ... + bN*xN
#
# With there being as many coefficients (b) as there are input features, plus 1 (for the b0 coefficient, or intercept).
#
# We don't know exactly how they play together for a given data set so we're going to use a mechanism to pick some random values, and gradually improve them over time.
#
# The method we discussed in class is called Stochastic Gradient Descent, and is one of a variety of related optimization algorithms that are used in many machine learning methods to 'learn' the coefficients, or weights, on input values with respect to some output.
#
# Gradient descent it the process of minimizing some function (we'll call it the cost, or error function) by following the slope (or gradient) of the function down to some minimum. Intuitively, we're going to show our model one training instance at a time, make a prediction for that instance, calculate the error using that instance, and update the model in order to improve performance (get a smaller error) for the nect prediction. We'll repeat this process for a number of iterations, minimizing the error each time.
#
# Each iteration, we're going to update the coefficients using the formula:
#
# b = b - learning_rate * error * x
#
# for each coefficient (again, corresponding to each input feature, of each instance).
#
# Remember also that learning_rate is a value we must choose (I'll tell you to start with), and the number of iterations is ALSO a number we must choose. Let's start.

# ### Part 1: Making Predictions
#
# (a) We're going to use predictions made by our model as a guide for tuning the coefficients, we need a function that can make predictions. This function can also apply the final coefficients we have learned to make predictions over new data. Write a function called predict(instance, coefficients), that takes a single instance (a list of input features), and a list of coefficients, and calculates the predicted y value, using the formula:
#
# y = b0 + b1*x1 + b2*x2 + ... + bN*xN
#
# This should work for ANY length of instance, but we can always assume that the length of the instance list and the length of the coefficent list are the SAME. For the instance, the values are:
#
# [x1,x2,x3,...,xN,Y] (where Y is the actual value, in the case of training, or None, in the case of testing).
#
# For the coefficients, let's assume that the list contains all the coefficients, including b0. Let's also assume that coefficients[0] is ALWAYS where we store b0.
#
# Your function predict(instance, coefficients) should return the predicted y value for a given instance and set of coefficients.
#
# (b) In the Simple Linear Regression assignment, you applied your model to a contrived data set. I've reproduced this below. Go through this data set one instance at a time and call your new predict function for each instance. You can use the coefficients [0.4, 0.8], which are almost exactly what you learned as coefficients in Assignment 2.
#
# For each instance, print out the correct value, and the value predicted by your function from (a). You should see that it performs reasonably well.

# In[ ]:

#Easy print method
def fprint(label, value) :
	print("{0}: {1}".format(label, value))



# Write your predict function here
def predict(instance, coefficients) :
	yVal=coefficients[0];

	for i in range(1, len(instance)) :
		mult=coefficients[i]*instance[i-1]
		yVal=yVal+mult;
	return yVal;



# Apply your function to the contrived dataset

dataset = [[1,1],[2,3],[4,3],[3,2],[5,5]]
coef = [0.4,0.8]

test=[[item[0],None] for item in dataset]

for i in range(len(dataset)):
	fprint("dataset {0}\nActual Value".format(i), dataset[i][1])
	fprint("predicted",predict(test[i],coef))



# ### Part 2: Learning coefficients with Stochastic Gradient Descent
#
# We now need to estimate coefficients for ourselves. For a given learning rate, for a number of iterations (epochs), we're going to estimate our coefficients.  Given a set of training data, we're going to:
#
# - loop over each epoch
# - loop over each row (instance) in the training data, for an epoch
# - loop over each coefficient, for each feature in each instanc, for each epoch
#
# As computer scientists you should recognize that this requires three, nested for loops, and you should have a sense (a big-O kind of sense) why this can take a long time for large data sets.
#
# Coefficients are updated based on the error the model made. The error is calculated as the difference between the predicted y value and the actual y value:
#
# error = prediction - actual
#
# There is one coefficient for EACH input attribute, and these are updated every time, for example:
#
# b1 = b1 - learning_rate * error * x1
#
# We ALSO need to update the special intercept coefficient b0:
#
# b0 = b0 - learning_rate * error
#
# (c) Implement the following algorithm for Stochastic Gradient Descent, naming your function coefficientsSGD(train, learning_rate, epochs), where train is the training data, and the other two parameters are the ones that control the learning.
#
# The algorithm is as follows:
#
# - initialize a list for the output coefficients. The length of the list will be the same as the length of every instance in the training data. We can initialize all the coefficients to 0.0 in the first instance
# - for each epoch
#     - initialize the total error to 0
#     - for each instance in the training data
#         - calculate the prediction for that instance, given the current list of coefficients, using our function from (a)
#         - calculate the error for that prediction
#         - (remember, each instance of the training data has the actual Y value)
#         - square the error, and add it to the total error. We're going to print the total error each time, and squaring the individual error means it will always be a positive value. NOTE: We don't use this squared error for updating the coefficients - we use the original error. This squaring is just to give us nice, readable output.
#         - Now update the coefficients, using the formulas given above. One update for b0 (which should always be at position 0 in the coefficients list), and then a series of updates for the remaining coefficients, b1 through bN.
#
#     - At the end of each epoch, print out the epoch number (we can start at epoch 0), the learning rate, and the total error for this epoch.
# - once we've iterated through each epoch, return the list of coefficients
#
# (d) Apply your coefficientsSGD function to the contrived dataset, given below. If it's working, you should see the error rate falling each epoch. You should also note that the value of the coefficients learned isn't quite the same as Simple Linear Regression, because we're estimating each time. You could try learning longer (more epochs), or altering the learning rate, and see if the coefficients approach the optimal values we learned in Assignment 2.

# In[ ]:


# Write your coefficientsSGD(train,learning_rate,epochs) here
def coefficientsSGD(train, learning_rate, epochs) :
	coefficents=[0.0 for i in range(len(train[0]))]
	for i in range(0, epochs) :
		error=0;
		for n in range(0, len(train)) :
			instance=train[n]
			predicted=predict(instance,coefficents)
			instErr=predicted-instance[-1:][0]

			error += math.sqrt(instErr**2)
			coefficents[0]=coefficents[0] - learning_rate * instErr
			for j in range(1,len(instance)) :
				coefficents[j]=coefficents[j] - learning_rate * instErr * instance[j-1]

		fprint("Epoch",i)
		fprint("Learning rate", learning_rate)
		fprint("Total Error", error)
		fprint("coefficents", coefficents)
	return coefficents

# Apply to the contrived data here. Try my parameters first, before you experiment

dataset = [[1,1],[2,3],[4,3],[3,2],[5,5]]

learning_rate = 0.001
epochs = 50
#coefs = coefficientsSGD(dataset, learning_rate, epochs)
#print('Coefficients are:',coefs)


# (e) Now you have sufficient functionality to write a function to make predictions using multivariate linear regression. Create a function with the signature mlr(train,test,learning_rate,epochs). Remember, training data is data containing the features of the data AND the class. Testing data contains the features, but does NOT contain a class (instead it should hold the value None in place of the class entry).
#
# We're going to use the same dataset here for both training and testing, even though we know that might not be a great idea.
#
# Here's the mlr algorithm. We're going to estimate our coefficients from the training data, using the function from (c) above. We're going to create a new list, to hold our predictions. Then for each entry in the testing data, we're going to read the input value, and make a prediction, using our function from (a). For each entry in the test data, we're going to append our predicted y value to the prediction list. We're going to return our list of predictions.

# In[ ]:


# Write function mlr(train,test,learning_rate,epochs) here
def mlr(train,test,learning_rate,epochs) :

	coefs = coefficientsSGD(train, learning_rate, epochs)
	predictions=[]
	for i in range(len(test)) :
		predictions.append([test[i], predict(test[i],coef)])

	return predictions


# ### Part 3: Applying to real data
#
# Last time I gave you the wine quality data set. Let's use that for this assignment. You'll need to:
#
# (f) Load the data set
# (g) Convert the features from strings to floats
# (h) Normalize all the attributes
# (i) Call the evaluate_algorithm function, given below with both your mlr algorithm, and a baseline algorithm (zeroRR). Print out the RMSE for both.
#
# Executing the above will require you to copy across a number of functions from the previous assignments.
# At the end, write something about the result. To do that, it will be helpful to interpret both the final RMSE from the multivariate linear regression, and to know something about the original features. The wine data set is just the white wine component of the data you can find here:
#
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality
#
#

# In[ ]:

# Write your code for f through i here
import csv
import matplotlib.pyplot as plt
import math

# CSV load method
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



wineData=load_data("winequality-white.csv")



# Write your converter function here
def column2Float(dataset,column) :
	dataset[column]=[float(data) for data in dataset[column]]



# Apply to the loaded Swedish auto data here
for i in range(len(wineData)) :
	column2Float(wineData,i)


# Print nicely here
#fprint("instances: {0} \nfeaturs: {1} \n data".format(len(wineData), len(wineData[0])), wineData)



#rmse_eval
def rmse_eval(actual,predicted) :
	result=[(pred[1]-act)**2 for act, pred in zip(actual, predicted)]
	sumation=sum(result)
	average=sumation/len(actual)
	return average

#ZeroRR
def zeroRR(train,test) :
	Y=0
	for item in train :
		Y+=item[-1:][0]
	meanVal=Y/len(test)
	mean=train
	return [[item[0], meanVal] for item in test]





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


def evaluate_algorithm(dataset, algorithm, metric, *args):
	train = dataset
	test = [[item[:-1],None] for item in train]
	predicted = algorithm(train,test,*args)
	actual =[val[1] for val in train]
	fprint("ACTUAL VALUE", actual)
	result = metric(actual,predicted)
	return result





# Testing multivariate linear regression


normalize(wineData, minmax(wineData))

dataset = wineData
learning_rate = 0.01
epochs = 50
mlr_result = evaluate_algorithm(dataset,mlr,rmse_eval,learning_rate,epochs)
zeroRR_result = evaluate_algorithm(dataset,zeroRR,rmse_eval)

print('MLR RMSE: %.3f' % mlr_result)
print('zeroRR RMSE: %.3f' % zeroRR_result)



# #### Part 2 Discussion Here
#I ran part 2 for 5000 epochs, and ended up with significantly closer coefficent values, which is a nice confimation, but doesn't teach anything new.
#
# Write something about the results here
#According to the rmse values of both MLR and ZeroRR, ZeroRR wins out on this dataset.
#As we know how both of these algorithms work, this initially seems like a good sign that something was done wrong, however further examination of the dataset itself shows that the "result"(the quality of wine) of the dataset displays an affinity towards average ratings (because it is white wine). As ZeroRR fits this trend very nicely, its rmse evaluation turns out better than MLR (at 50 and 5000 epochs). theirfore, as we learned in class, the perfomance of an algorithm on a dataset means little without the full context.
#
