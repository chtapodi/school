#!/usr/bin/env python
# coding: utf-8


# # CSC-321: Data Mining and Machine Learning
# ## Assignment 3: Working with real data
# 
# ### Part 1: Loading Data
# 
# We've been working with contrived data, but it's time to work with 'real' data. The data we will use will be in CSV format, and we'll know that it'll require minimum wrangling - but there are still things we will need to do. 
# 
# We're going to use the built-in python module csv to read our file. It will take a csv file and return a data structure that contains lists of list. where each list will be an instance of our data. However, those lists will contain strings, and we'll need to convert them to floats, on a column by column basis. 
# 
# There are two data sets on the homework page: the wine quality dataset and the Swedish insurance data set. Download them, and put them in the same directory as this notebook.


# 
# (a) Read the overview of the csv module at: https://docs.python.org/3.1/library/csv.html
# 
# Write a function called load_data(filename) that takes a filename (a string), uses the csv reader to read in a file, 
#then iterates through that file one line at a time and appends it to a new list, which is then returned. 
# 
# (b) Call this function on the Swedish auto data set given, and use the returned information to print a nice string. 
#That string should tell us:
# 
# The name of the data set, the number of instances (that's the number of sublists), and the number of features 
#(the columns in each sublist). 
# 
# (c) Print the first instance contained in the data set.
# 
# For example, if I were to load the contrived data set given last week from csv I should load the data as:  
# 
# [['1','1'],['2','3'],['4','3'],['3','2'],['5','5']] 
# 
# and report that there are 5 instances, with 2 features. I should then print:
# 
# [['1','1']]
# 
# Yes, at present, using this mechanism, the features will all be strings. We'll correct this later.

# In[ ]:

print("PART I")

import csv
import matplotlib.pyplot as plt
import math



#Easy print method
def fprint(label, value) :
	print("{0}: {1}".format(label, value))

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

				
# Print nicely
carData=load_data("insurance.csv")
fprint("instances: {0} \nfeaturs: {1} \ncar data".format(len(carData), len(carData[0])), carData)

fprint("first instance", carData[0])


# (c) Now we need to convert the strings to floats. That might not always be true, so we want a way to select certain 
#columns of our data, and turn the values into floats.
# 
# Write a function called column2Float(dataset,column), that takes a dataset and a column number, and converts the 
#elements in that column to from strings to floats. So if the data set contained:
# 
# [['1','1'],['2','3'],['4','3'],['3','2'],['5','5']]
# 
# and I called column2Float(dataset, 0), the dataset should be changed to be:
# 
# [[1.0,'1'],[2.0,'3'],[4.0,'3'],[3.0,'2'],[5.0,'5']]
# 
# Note. For memory efficiency, it's ok at this stage to make use of the the pass-by-reference nature of lists - that is, 
#it's ok to make these changes in place, and NOT create or return another list.
# 
# (d) Call your function string2Float on the swedish auto data set you loaded previously. Use a loop to iterate through 
#each of the columns, changing each into float values.
# 
# (e) Print the first line of the newly converted data set, to show that the entries are now floats.

# In[ ]:


# Write your converter function here
def column2Float(dataset,column) :
	dataset[column]=[float(data) for data in dataset[column]]



# Apply to the loaded Swedish auto data here
for i in range(len(carData)) :
	column2Float(carData,i)


# Print nicely here
fprint("first instance, updated", carData[0])
print("PART II")

# ### Part 2: Applying to real data
# 
# You probably noticed that the Swedish auto data set you loaded has only 2 features, which correspond to one input 

#Copy across your code for simple linear regression from last week, and 
#run it on this data. I want to see:
# 
# (a) A plot of x,y of the Swedish auto data
# (b) Some nice print out of a baseline of performance (What algorithm should you use?)
# (c) Some nice print out of the performance of simple linear regression
# (d) A line plot of the predicted y values made by simple linear regression
# (e) Some discussion of the results. This will NOT be extensive. Which algorithm (apparently) performs better, 
#the baseline or slr? What do the results MEAN, if anything?
# 
# You can find minimal information about the Swedish auto insurance data set here:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr06.html

# In[ ]:


#a
plt.subplot(2, 1, 1)
plt.title('Auto Data')
plt.plot(*zip(*carData), 'b^')
#plt.show()


# Write your code for a through d here
def mean(listOfValues) :
	return sum(listOfValues)/len(listOfValues)


def variance(listOfValues,meanValue) :
	list = [(val-meanValue)**2 for val in listOfValues]
	return sum(list)

def covariance(listofX, meanOfX, listOfY, meanOfY) :
	return sum([(xVal-meanOfX)*(yVal-meanOfY) for xVal,yVal in zip(listofX,listOfY)])
	
def coefficients(dataset) :
	x,y=zip(*dataset)
	meanX=mean(x)

	b1=covariance(x,meanX,y,mean(y))/variance(x,meanX)
	b0=mean(y)-b1*meanX
	return b0,b1




def slg(train,test) :
	b0,b1= coefficients(train)
	y=[b0+b1*item[0] for item in test]
	
	return [[test[n][0],itemy] for (n,itemy) in enumerate(y)]


# Call the function slg below
test=[[item[0],None] for item in carData]
processed=slg(carData, test)



# Plot the results
plt.subplot(2, 1, 2)
plt.title('Given Data + SLR')
plt.plot(*zip(*carData), 'b^')
plt.plot(*zip(*processed), 'r^')


# Write the function zeroRR(train,test) here
def zeroRR(train,test) :
	xTrain,yTrain=zip(*train)
	meanVal=mean(yTrain)
	return [[item[0], meanVal] for item in test]


# Call the function zeroRR below
predicted=zeroRR(carData,carData)


fprint("zeroRR baseline",predicted)
fprint("Linear Regression",processed)






# #### Part 2 Discussion Here
# 
#I used ZeroRR as baseline, as this is a good baseline to use.
#Simply printing out the value makes SLR look significantly better than ZeroRR, but the plot brings back to mind the fact that it is just a line, and theirfore fits the data nearly as minimally as possible.

#
# 
# 

# ### Part 3: Normalization
# 
# When working with data that has multiple inputs, we often want to normalize the data, so that it's all on the same scale 
#(usually 0-1). The steps to do that are below. 
# 
# (a) Write a function minmax(dataset) that goes through your data set, and returns a list of lists. Each sublist should 
#contain the min and the max of each column in your data. 
# 
# (b) Write a function called normalize(dataset, minmax). That code should go through each row in your data set 
#(each instance), and normalize each value. The argument minmax should be the contents of a list as in part (f), above. 
#The function to normalize a value, if you know the min and the max values of the column of data in which the value 
#appears is:
# 
# normalized value = value - minOfColumn / maxOfColumn - minOfColumn
# 
# For testing, I give you the data set below, and in comments, the output of both the minmax function, and the 
#resulting normalized data set so you can check your function
# 
# (c) Run your code on the wine quality data set. You will need to:
# - Load it
# - Convert it to floats
# - Normalize it
# When complete, print the first line of data at each step - so I should see a line corresponding to when the data 
#was loaded, a line after conversion to floats, and a line after normalization.

# In[ ]:


# Write minmax and normalize here
def minmax(dataset) :
	cols=dataset[0]
	cols=[[val,val] for val in cols]
	for data in dataset :
		for i in range(0,len(cols)) :
			#fprint("data {0}".format(i),data[i])
			#fprint("cols {0}".format(i),cols[i])
			current=data[i]
			if current<cols[i][0] : #if current column data is less than stored, replace
				cols[i][0]=current
			if current>cols[i][1] :
				cols[i][1]=current
	return cols
	
	
def normalize(dataset, minmax) :
	if (len(dataset[0])!=len(minmax)) :
		print("lengths do not match")
		print("{0} !={1}".format(len(dataset[0]),len(minmax)))
		exit()
	
	for data in dataset :
		for i in range(0,len(minmax)) :
			#fprint("i:{0}, data".format(i),data[i])
			data[i]= (data[i] - minmax[i][0] )/( minmax[i][1] - minmax[i][0])

	




# Test on the following

dataset = [[50, 30, 100], 
           [20, 90, 27], 
           [100, 45, 63], 
           [400, 25, 19]]
		 
fprint("minmax", minmax(dataset))

normalize(dataset, minmax(dataset))
fprint("dataset", dataset)

# minmax should return: [[20, 400], [25, 90], [19, 100]]
# normalized data should be:
# [[0.07894736842105263, 0.07692307692307693, 1.0], [0.0, 1.0, 0.09876543209876543], [0.21052631578947367, 0.3076923076923077, 0.5432098765432098], [1.0, 0.0, 0.0]]


# Then test on the wine quality dataset


# In[ ]:

wineData=load_data("winequality-white.csv")
fprint("just loaded", wineData[0])

for i in range(len(wineData)) :
	column2Float(wineData,i)
	
fprint("converted", wineData[0])

normalize(wineData, minmax(wineData))
fprint("normalized", wineData[0])

plt.show()


