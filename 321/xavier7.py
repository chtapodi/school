#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
# # Your Name
# ## Assignment 7: Classification with probability
#
# ### Part 1: Naive Bayes
#
# Everything so far has been a linear classifier. Now we'll move up a gear, and implement some non-linear classifiers. The first, as we saw in class, is Naive Bayes, that makes use of proability to make predictions.
#
# We make use of Bayes Theorem, that allows us to calculate the probability of a piece of data belonging to a given class, given our prior knowledge. Bayes Theorem is stated as:
#
# P(class|data) = (P(data|class) * P(class)) / P(data)
#
# Where P(class|data) is the probability of class given the provided data
#
# We're going to break this down into several steps. Again, I've given you a contrived data set for you to test your functions.

# #### (a) Separate by class
#
# Just as in class, we need to calculate the probability of data by the class they belong to. We'll need to separate our data by the class. Create a dictionary, where the key is class, and the values is a list of all instances with that class value.

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
# VERBOSE=True

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

def zeroRC(train, test) :
	yList=[inst[-1:][0] for inst in train]
	toReturn=[max(set(yList), key=yList.count) for i in test]
	fprint("toReturn", toReturn)
	return toReturn

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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Contrived data set

dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]


# implement separateByClass(dataset) here

def separateByClass(dataset) :
	dict = {}
	for item in dataset :
		# print(item[-1:][0])
		dict.setdefault(item[-1:][0], []).append(item)
	return dict




# #### (b) Summarize the data
#
# We need two statistics from the data, the mean and the standard deviation. You should have these functions in a previous assignment, remembering the standard deviation is simply the square root of the variance.

#We need the mean and standard deviation for each of our attributes, i.e. for each column of our data.

#Create a function that summarizes a given data set, by gathering all of the information for each column, and calculating the mean and standard deviation on that columns data.
#We'll collect this information into a tuple, one per column, comprising the mean, the standard deviation and the number of elements in each column).
#Return a list of these tuples.

# In[ ]:

def mean(listOfValues) :
	return sum(listOfValues)/len(listOfValues)


def variance(listOfValues,meanValue) :
	list = [(val-meanValue)**2 for val in listOfValues]
	return sum(list)

def standard_dev(listOfValues) :
	var=variance(listOfValues, mean(listOfValues))
	return math.sqrt(var/(len(listOfValues)-1))

# implement summarizeDataset(dataset) here, and copy across any functions you need to help you

def summarizeDataset(dataset) :
	toReturn=[]
	colList=list(zip(*dataset))
	for col in colList[:-1] :
		toReturn.append((mean(col), standard_dev(col), len(col)))
	return toReturn





# #### (c) Summarize data by class
#
# We now need to combine the functions from (a) and (b) above. Create a summarizeByClass function, that splits the data by class, and then caluclates statistics for each row of the data for each class. The results - the list of tuples of statistics, one per column - should then be stored in a dictionary by their class value. summarizeByClass should return such a dictionary.

# In[ ]:



# implement summarizeByClass(dataset) here

def summarizeByClass(dataset) :
	dict=separateByClass(dataset)
	for key, val in dict.items() :
		dict[key]=summarizeDataset(val)

	return dict

print(summarizeByClass(dataset))


#"Right. Yeah, I forgot to mention that we're taking the standard deviation of a SAMPLE, not a POPULATION, so we have to correct for that. If you remember, to calculate the average variance (which we need here), you divide the squared error by something. For a population (like the dogs in the class example) that's the length of the population. For a sample (which we'll be dealing with most of the time, it's the length of the sample - 1."



# The dictionary for the contrived data should look like:
# {0: [(2.7420144012, 0.9265683289298018, 5), (3.0054686692, 1.1073295894898725, 5)], 1: [(7.6146523718, 1.2344321550313704, 5), (2.9914679790000003, 1.4541931384601618, 5)]}


# #### (d) Guassiaun Probability Density
#
# We're working with numerical data here, so we need to implement the gaussian probability density function (PDF) we talked about in class, so we can attach probabilities to real values. A gaussian distribution can be summarized from two values - guess which two? If you guessed mean and standard deviation, you were correct. The gaussian PDF is calculated as follows:
#
# probability(x) = (1 / (sqrt(2 * pi) * std_dev)) * exp(-((x-mean) ** 2 / 2 * std_dev ** 2 )))
#
# Hopefully, you can see why we're going to need the mean and the std_dev from function (c)
#
# Create a function that:
# - takes a value
# - takes a mean
# - takes a standard deviation
#
# and returns the probability of seeing that value, given that distribution, using the formula above.

# In[ ]:



# Implement calcProb(value, mean, std_dev) here
def calcProb(value, mean, std_dev) :
	return (1 / (math.sqrt(2 * math.pi) * std_dev)) * math.exp(-((value-mean) ** 2 / (2 * std_dev ** 2 )))





# #### (e) Class Probabilities
#
# We can now use probabilites calculated from our training data to calculate probabilities for an instance of new data, by creating a function called calcClassProbs. Probabilites have to be calculated separately for each possible class in our data, so for each class we have to calculate the likelihood the new instance of data belongs to that class. The probability that a piece of data belongs to a class is calculated by:
#
# p(class|data) = p(X|class) * P(class)
#
# The divison has been removed, because we're just trying to maximize the result of the formula above. The largest value we get for each class above determines which class we assign. Each input value is treated separately, so in the case where we have TWO input values in our data (X1 and X2), the probablility that an instance belongs to class 0 is calculated by:
#
# P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0)
#
# We have to repeat this for each class, and then choose the class with the highest score. We should not assume a fixed number of input features, X, the above was just an illustration.


#So based on each class in the instance, muliply the probability based on the class


# We'll start by creating a function that will return the probabilities of predicting each class for a given instance. This function will take a dictionary of summaries (as returned by (c), above) and an instance, and will generate a dictionary of probabilites, with one entry per class. The steps are as follows:


#TODO
#0[n][-1]+1[n][-1]
# - We need to calculate the total number of training instances, by counting the counts stored in the summary statistics. So if there are 9 instances with one label, and 5 with another (as in the weather data) then we need to know there are 14 instances.
#
# - This will help us calculate the probability of a given class, the prior probability P(class), as the ratio of rows with a given class divided by all rows in the training data
#
# - Next probabilities are calculated for each input value in the instance, using the gaussian PDF, and the statistics for that column and of that class. Probabilites are multiplied together as they are accumulated with the formula given above.
#
# - The process is repeated for each class in the data
#
# - Return the dictionary of probabilities for each class for the new instance
#
# Some things that might help with implementation.
#
# - Dictionaries are your friend here
# - The data returned by (c) above is already divided by class. You can:
#     - discover the prior probability from this data (how many instances for this class, divided by the total instances)
#     - iterate over the tuples, which give you the information (mean, std_dev, count) on a per column basis
#     - calculate probability given the attribute value corresponding to that column using your function from (d)
#
# Try this out on the contrived data.
#
# NOTE: If you want to output ACTUAL probabilities by class, we divide each score in the dictionary for an instance, by the sum of the values. You don't need to do this, it's just a reminder.
#

# In[ ]:



# Implement calcClassProbs(summaries, instance) here

def calcClassProbs(summaries, instance) :
	fprint("summaries in", summaries)
	fprint("instance in", instance)
	#sums up number of instances
	numInst=0
	for classes in summaries.values() :
		for stat in classes :
			numInst+=stat[len(stat)-1]

	#get P(class)

	for key in summaries.keys() :
		keyInstances=0
		statList=summaries[key]
		for stats in statList :
			keyInstances+=stats[2]
		priorProb=keyInstances/numInst

		# fprint("priorProb",priorProb)
		probMult=priorProb
		for i in range(len(instance)-1) : #iterate through Xvals
			#Notes
			#stats=(mean(col), standard_dev(col), len(col)))
			#calcProb(value, mean, std_dev) PDF

			#Naming
			mean=statList[i][0]
			std_dev=statList[i][1]
			instProb=calcProb(instance[i], mean, std_dev)

			# fprint("key",key)
			# fprint("probability of",instance[i])
			# fprint("in statlist",statList[i])
			# fprint("is",instProb)
			# print("\n")

			probMult=probMult*instProb


		summaries[key]=probMult


	return summaries

# fprint("totProb",calcClassProbs(summarizeByClass(dataset), dataset[0]))





# Test it out here


#UNCOMMENT

#
# summaries = summarizeByClass(dataset)
# probabilities = calcClassProbs(summaries, dataset[0])
# print('Probabilities are:',probabilities)

# I think if everything works, it should be:
# {0: 0.05032427673372075, 1: 0.00011557718379945765}
# which according to the percentage calculation give above should be:
# 99.77% in favour of class 0

# sumProbs = sum([v for _,v in probabilities.items()])
# for k,v in probabilities.items():
# 	print('The probability of the instance belonging to class %d is %.2f' % (k,v/sumProbs*100))


# #### (f) Tying it all together
#
# You need to create a predict function. This function works very much as the example above, in that it takes a dictionary of summaries and a single row, and uses calcClassProbabilites to get the dictionary of probabilities. From this dictionary, find the largest value and corresponding class. Return this class.
#
# You also need a naiveBayes function, that takes a training set and a test set. It needs to generate summary statistics from the training set (using (c), above), then make predictions for each instance in the test set, by calling your predict function above for each instance, using the summaries generated. Append these predictions to a list you return.

# In[ ]:



# Implement predict(summaries,instance) here

def predict(sumStats,instance) :
	fprint("predict: sumStats", sumStats)
	probabilities = calcClassProbs(sumStats, instance)
	return max(probabilities, key=probabilities.get)





# Implement naiveBayes(train,test) here
def naiveBayes(train,test) :
	summarizedStats=summarizeByClass(train)
	fprint("naiveBayes: InitStats", summarizedStats)
	predicted=[]
	for instance in test :
		fprint("naiveBayes: sumStats", summarizedStats)
		predicted.append(predict(copy.deepcopy(summarizedStats), instance))
	return predicted






# ### Applying to real data
#
# You've seen bits of the iris dataset in class. It's one of the most well known data sets in machine learning and data mining. So you might as well have a go at it! You can find out more about it here: http://archive.ics.uci.edu/ml/datasets/Iris
#
# You'll need to:
#
# - Load the data
# - convert all but the last column to floats
# - convert the last column to an int. There are THREE classes, so convert them to 0, 1 and 2 accordingly
# - call evaluate algorithm, using a 5-fold cross-validation
# - print the mean, min and max scores
# - compare this to some reasonable baseline
# - give me a very short write up of the results

# In[ ]:
filename = 'iris.csv'

loadedData=load_data(filename)
loadedData=str(loadedData) #cast to string
loadedData=loadedData.replace('Iris-setosa', '0') #replace using strings built in method
loadedData=loadedData.replace('Iris-versicolor', '1')
loadedData=loadedData.replace('Iris-virginica', '2')
loadedData=ast.literal_eval(loadedData) #convert back to nested list

#convert all values to float (You said make the last int, but I don't see why float won't work. Hopefully I'm not overlooking something)
for i in range(len(loadedData)) :
	column2Float(loadedData,i)
dataset=loadedData

print("{0}:\ninstances: {1} \nfeaturs: {2}".format(filename, len(loadedData), len(loadedData[0])))

folds = 5


nBayes=evaluate_algorithm(dataset, naiveBayes, folds, accuracy)
zeroRCScore = evaluate_algorithm(dataset, zeroRC, folds, accuracy)


VERBOSE=True
fprint('Naivebayes', nBayes)
fprint('   best', max(nBayes))
fprint('  worst', min(nBayes))
fprint('   mean', stat.mean(nBayes))

fprint('zeroRC', zeroRCScore)
fprint('   best', max(zeroRCScore))
fprint('  worst', min(zeroRCScore))
fprint('   mean', stat.mean(zeroRCScore))

# Write up your observations on the experiment here
#Based on the accuracy evaluation algorithm, the worst Naivebayes score is 3-times greater than the best ZeroRC score. As ZeroRC is not the most accurate algorithm, this by itself does not mean much, but the mean Naive Bayes Accuracy for this data is 95.33333333333333, meaning that on average this model performed with less than 5% error, which is hard to argue as a bad thing in these circumstances.
#
