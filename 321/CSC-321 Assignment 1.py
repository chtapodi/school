#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
# ## Assignment 1: List comprehension and numpy
# 
# ### Part 1: List comprehension, list indexing
# First, read the tutorial at: https://www.programiz.com/python-programming/list-comprehension
# 
# Take the following list, below, and use list comprehension to complete the following. You must use list comprehension, and you mmust start with the original list (startList) each time. 
# 
# (a) Create a new list from every other element, starting at index 0
# 
# (b) Create a new list that contains True for each element in the original list that is even, False if it is odd
# 
# (c) Create a new list that replaces the values in the original list with 1, 2 or 3 under the following conditions. If the value is less than 3, make the new value 1, if it's greater than or equal to 3. but below 6, make it equal to 2, otherwise make it 3
# 
# (d) Use slicing to make a shallow copy of the complete list. To test, change one element of your new list, and print both lists. Make sure ONLY your new list changes (because lists are pass-by-reference)
# 
# (e) Go through the list of lists contained in the variable data, and extract all elements from each list EXCEPT the last one, and put into a new list of lists. You must use list comprehension.
# 
# e.g. the output should be: [[1, 2], [4, 5], [7, 8], [10, 11]]
# 
# (f) Go through the list of lists contained in the variable data, and create a single list containing ONLY the last element of each nested list. You must use list comprehension.
# 
# e.g. the output should be: [3, 6, 9, 12]

# In[ ]:


# Use this list for a-d above

startList = [2, 7, 1, 9, 1, 4, 8, 10, 2, 3, 5]
print("original {0}".format(startList))

a=[item for n, item in enumerate(startList) if n%2==0]
print("a {0}".format(a))

b=[True if item%2==0 else False for item in startList]
print("b {0}".format(b))

c=[1 if item<3 else 2 if (3<=item<6) else 3 for item in startList]
print("c {0}".format(c))

d=[startList[n:n+1][0] for n, item in enumerate(startList)] #Did you want us to use list comprehension and slicing?

print("shallow copy {0}".format(d))

d[0]=1
print("changed copy {0}".format(d))
print("original {0}".format(startList))



# Use this list for e-f above

data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
print("data {0}".format(data))

e=[item[:-1] for item in data]
print("e {0}".format(e))

f=[item[-1:][0] for item in data]
print("f {0}".format(f))

# ### Part 2: Multiple return values
# 
# Read the tutorial at: https://www.tutorialspoint.com/How-do-we-return-multiple-values-in-Python
# 
# (a) Write a function called minmax, that takes a list as a single argument, and returns two values as a tuple - the minimum value in the list, and the maximum value in a list. 
# 
# (b) Use the random.randint(a,b) function to create a list filled with 100 random integers in the range 0 to 100 (inclusive).
# 
# (c) Pass this list to your function from (a), capture the two return values into TWO distinct variables, and print them nicely! (i.e. Print two, distinct, meaningful output messages, not just the tuple returned).

# In[ ]:

def minmax(inList) :
	return min(inList), max(inList)
	
import random

randList=[random.randint(0,100) for i in range(100)]
print(randList)

returnedData=minmax(randList)
print("distinct meaningful output value 1: {0}\ndistinct meaningful output value 2: {1}".format(returnedData[0],returnedData[1]))


# ### Part 3: Numpy
# 
# Read the tutorial at: https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# You may also want to check the following for information on array indexing in more detail: https://www.tutorialspoint.com/numpy/numpy_indexing_and_slicing.htm
# 
# Then work on the following:
# 
# (a) convert the startList to a one dimensional array. Print this array, and it's shape. Nicely.
# 
# (b) convert the startList into a two dimensional array, of 4 rows of 3 columns. Print this array, and it's shape. Nicely.
# 
# (c) use array slicing on (b) to produce similar output to 1(e) and 1(f) above. One variable containing 4 rows of 2 columns (all the data EXCEPT the last column from each row), 
#and one variable that is a 4 by 1 array, containing the values from the last column. You will need to reshape this last variable (so a (4,) shape is not acceptable, but a (4,1) is). 

# In[ ]:


import numpy as np

startList = [12,26,18,1,15,18,12,19,2,7,4,9]

startArray=np.array(startList)
print("array'd list {0}".format(startArray))
print("array'd list shape {0}".format(startArray.shape))

array2D=startArray.reshape(4, 3)
print("2D array {0}".format(array2D))
print("2D array shape {0}".format(array2D.shape))

print("problem 3c(1(e)) {0}".format(array2D[:, :-1]))

print("problem 3c(1(e)) {0}".format(array2D[:, -1].reshape(4,1)))
