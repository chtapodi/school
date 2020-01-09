#!/usr/bin/env python
# coding: utf-8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ## Author: Xavier Theo Quinn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Imports
import csv
import matplotlib.pyplot as plt
import math
import copy
import random

from string import ascii_lowercase
from datetime import datetime
from random import randint

import numpy as np
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


def writeCSV(filename, csv_list) :
	#Write the data to a csv so I don't need to melt my computer every time I need this data
	with open(filename, 'w', newline='') as myfile:
		 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		 wr.writerows(csv_list)

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#PERMINANTLY BORROWED CODE


#Taken from this guy on stack overflow: https://codereview.stackexchange.com/questions/183658/replacing-letters-with-numbers-with-its-position-in-alphabet
LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)}
def alphabet_position(text):
	text = text.lower()

	numbers = [LETTERS[character] for character in text if character in LETTERS]

	return ' '.join(numbers)


#https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)

def scatterPlot(dataSet):
	N = len(dataSet)
	cmap = get_cmap(N)
	for i in range(N):
		x=[]
		y=[]
		cList=[]
		for entry in dataSet[i] :
			x.append(entry[2][0])
			y.append(entry[2][1])
			cList.append(cmap(i))

		plt.scatter(x,y, c=cList)
	plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getEucDist(posOne, posTwo) :
	# return distance.euclidean(posOne, posTwo) #TODO
	dist=math.sqrt((posOne[0]-posTwo[0])**2+ (posOne[1]-posTwo[1])**2)
	# fprint("Dist", dist)
	return dist

#returns a list of lists of temporally close data
def chunkByTime(dataSet, minTime) :
	toReturn=[]
	chunk=[]
	t0=dataSet[0][0]
	for i in range(0,len(dataSet)) :

		timeDiff=(abs(dataSet[i][0]-t0)).total_seconds()
		if (timeDiff<=minTime) :
			# fprint("timeDiff",timeDiff)
			# print(dataSet[i])
			chunk.append(dataSet[i])
		else :
			t0=dataSet[i][0]
			toReturn.append(chunk)
			chunk=[]
	return toReturn

#This iterates through the dataset and groups all points based on distance to eachother
def chunkByDist(dataSet, minDist) :
	dataStack=copy.deepcopy(dataSet)
	toReturn=[]
	# fprint("MinDist", minDist)
	# fprint("len", len(dataStack))

	while len(dataStack)>0 :
		point=dataStack.pop(0)
		d0=point[2]
		chunk=[point]
		rejects=[]
		i=0
		for entry in dataStack :

			d1=entry[2]
			dist=getEucDist(d0,d1)

			"  {0}".format(i)

			if dist<=minDist :
				chunk.append(entry)
				"X {0}".format(i)
			else :
				rejects.append(entry)
		toReturn.append(chunk)
	return toReturn


def sizeLimit(dataSet, limit) :
	toReturn=[]
	for entry in dataSet :
		if len(entry)>=limit :
			toReturn.append(entry)
	return toReturn

#returns the SIZE of the biggest chunk
def biggestChunk(dataSet) :
	biggestChunk=0
	#sort by datetime
	for entry in closeRssi :
		if len(entry)>biggestChunk :
			biggestChunk=len(entry)
		# fprint("Chunk of {}".format(len(entry)),entry)
	return biggestChunk


def flatten_rssi(rssi_list) :
	# print("start")
	# lprint("rssi_list",rssi_list)
	sel_list=rssi_list[0]
	remaining_lists=rssi_list[1:]
	# fprint("sel_list",sel_list)
	# fprint("remaining_lists",remaining_lists)
	for entry in remaining_lists :
		for i in range(len(sel_list)) :
			if (sel_list[i]==-200) :
				sel_list[i]=entry[i] #if its -200, it dont matter, if its not, it doesn't matter either.

			else :
				if (entry[i]!=-200) : #they are both not null
					av=(entry[i]+sel_list[i])/2
					sel_list[i]=av
				 #if it is 200, it just stays the same

	return sel_list



#returns a list of "flattened" chunks
#All beacon data is averaged as a column
def flattenChunks(dataSet) :

	passed=[]
	for entry in dataSet :
		time, rssi, loc = zip(*entry)
		#This is just some strait up fuckery
		#I should do it basically any other way.
		flatRssi=flatten_rssi(rssi)
		# for entry in rssi :
		# 	entry=[np.nan if x==-200 else x for x in entry]
		#
		# np_rssi=np.array(rssi) #flatten
		# np_rssi=nanmean(np_rssi,axis=0)
		# flatRssi=np.nan_to_num(np_rssi).tolist() #remove nans

		flatLoc=np.array(loc).mean(axis=0).tolist()
		flatTime=(max(time)-min(time))/2+min(time)
		length=len(flatRssi)
		badBeacons=flatRssi.count(0)
		if (length-badBeacons>=3) :
			# fprint("  flattened", flattened)
			passed.append([flatTime,flatRssi, flatLoc])
	# fprint("passed", passed)
	return passed



	for data in chunkedData :
		print("gap")
		for variableName in data :
			fprint("en",variableName)

#Returns one minus twp
def LLsubtract(one, two) :
	return [x for x in one if x not in two]



#Gets how many unique non -200 values there are between two lists
def getIndexVariety(one, two) :
	averaged=np.array([one,two]).mean(axis=0).tolist()
	variety=len(averaged)-averaged.count(-200)
	return variety
'''
Average arrays
+200 across whole array
count non 0 values
if its increased, do a thing
'''
def groupByN(dataSet, varReq) :
	referenceSet=copy.deepcopy(dataSet)
	clusterList=[]
	while len(referenceSet)>0 :
		leader=referenceSet.pop(randint(0,len(referenceSet)-1))
		theRest=copy.deepcopy(referenceSet)
		varience=getIndexVariety(leader[1],leader[1]) #gets varience for just this one
		# fprint("LEADER:{0}".format(varience),leader[1])
		cluster=[leader]
		while len(theRest)>0 and varience<varReq :
			toComp=theRest.pop(random.randint(0,len(theRest)-1))
			testVarience=getIndexVariety(leader[1],toComp[1])
			# fprint("toComp:{0}".format(testVarience),toComp[1])
			if  testVarience>varience : #if grouping will increase the varience
				cluster.append(toComp)
				varience=testVarience
				# fprint("  TRUE:{0}".format(varience),toComp[1])
		#Enough instances have been clustered to meet max varience
		if (varience>=varReq) :
			referenceSet=LLsubtract(referenceSet, cluster)
			clusterList.append(cluster)
	return clusterList




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#This is based off the provided map, and each index corosponds to the beacon at this locaiton in the provided data.
#1=a, etc...
beacon_locations=[(5,9),(10,4),(14,4),(19,4),(10,7),(14,7),(19,7),(10,10),(4,15),(10,15),(14,15),(19,15),(23,15)]
#generating a grid that represents the room
room_grid=[]
for i in range(18) :
	room_grid.append([0]*23)


#Load and clean data

filename="dataset.csv"
data=load_data(filename)
rssiData=[]

for entry in data[1:] : #start at 1 because the first line in CSV is column names

	#Converts alphanumeric position to int vector
	position=[int(entry[0][1:]),int(alphabet_position(entry[0][:1]))]

	fprint("position", position)
	#converts timestamp to usable format (I probably won't use this, but might as well have it)
	timeStamp = datetime.strptime(entry[1], '%m-%d-%Y %H:%M:%S')
	# print(timeStamp)

	#converts RSSI values into ints
	rssi=[float(data.strip()) for data in entry[2:]]
	# rssi=[None if x==-200 else x for x in rssi]
	#I can update this later, but as position is what we are trying to find, it follows convention to have it last.
	rssiData.append([timeStamp, rssi, position])



# writeCSV("cleaned_dataset.csv",rssiData)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #Goes through and makes a list of all entries with greater than a cutoff of valid dataPoints
# goodData=[]
# cutoff=3
# for entry in rssiData :
# 	positions=entry[1]
# 	if ((len(positions)-positions.count(None))>cutoff) :
# 		goodData.append(positions)
# 	# print(positions.count(-200))
#
# fprint(" goodData", goodData)
# fprint("#goodData", len(goodData))
# #So theres only 13 entries with >3 chunks of data.
# #This is a massive drawback
#
# sortedArray = sorted(rssiData,key=lambda x: x[0])
#
# # groups datapoints based on distance to eachother, which makes a lot of sense
# # closeRssi=chunkByDist(rssiData, 2)
#
#
# # fprint("Number of chunks",len(closeRssi))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #testing the filter algorithm
# # print(closeRssi[0][0])
# print("START")
# # chunked=chunkByDist(rssiData,15)
# chunked=[[[-70,-200,-200,-200,-200,-200,-200,-200,-200,-200,-200,-200,-200],[-200,-70,-200,-200,-200,-200,-200,-200,-200,-200,-200,-200,-200],[-200,-200,-70,-200,-200,-200,-200,-200,-200,-200,-200,-200,-200]],[[-200,-200,-200,-70,-70,-200,-200,-200,-200,-200,-200,-200,-200],[-200,-200,-200,-200,-200,-70,-200,-200,-200,-200,-200,-200,-200]],[[-200,-200,-200,-200,-200,-200,-200,-70,-70,-200,-200,-200,-200]]]
# # fprint("chunked",chunked)
# # fprint("filter",filterChunks(chunked))
# #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #maximize chunk size efficiency
# scoreList=[]
# evaluation=[]
# for i in range(0,27) :#as x-axis size is 21 and y axis is 18, 27 is the largest diagonal distance possible. Because values are only ints, distance of 0 is possible.
#
# 	chunkedData=chunkByDist(rssiData,i)
#
# 	evaluation=filterChunks(chunkedData)
# 	#
# 	# for entry in chunkedData :
# 	# 	# for val in entry :
# 	# 	print(entry)
#
#
# 	#As I am trying to get the most out of my dataset, I want the largest number of groups that exhibit the minimum beacon signals.
# 	#To do this I decided that by getting the ratio of
# 	score=len(evaluation) #This is kind of hacky, and I should fix it, but its just a cheap way to get how many chunks with >3 there are.
#
# 	scoreList.append([i,score])
# print(scoreList)
# plt.plot(*zip(*scoreList), 'b^-')
# # plt.show()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Testing clustering and flattening of specific varience
#
# chunkedData=chunkByDist(rssiData,5)
#
#
# clusteredTest=groupByN(chunkedData[1],3)
#
# flattenTest=flattenChunks(clusteredTest)
# for i in flattenTest :
# 	fprint("FLAT", i[1])
#
# for data in clusteredTest :
# 	print("gap")
# 	for variableName in data :
# 		fprint("en",variableName[1])
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This generates as many successful instances as possible
# lprint("rssiData",rssiData)

# flattenedList=[]
# group_list=[]
# chunkedData=chunkByDist(rssiData,5)
# for chunk in chunkedData :
# 	grouped=groupByN(chunk,8)
# 	if len(grouped)>0 :
# 		# fprint("GROUP", grouped)
# 		group_list.append(grouped)
# 		flattened=flattenChunks(grouped)
# 		# fprint("FLAT", flattened)
# 		flattenedList.append(flattened)
# # lprint("flat",flattenedList)
# fprint("#INSTANCES", len(flattenedList))
# #1373 instances!

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Now I need to organize this data based on how I will use it.
'''
I think the way to do this is to format the instances like:
[[(location,rssi),(location,rssi),(location,rssi),...],(signal_location)]
'''
# organized_data=[]
# for instance in flattenedList :
# 	beacon_list=[]
# 	for i in range(len(instance[0][1])) : #Iterates through beacon vals
# 		signal=instance[0][1][i]
# 		if signal!=-200 :
# 			beacon_list.append((beacon_locations[i],signal))
# 	organized_data.append([beacon_list,instance[0][2]]) #append the beacon list and the device location

# lprint("organized_data", organized_data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# writeCSV("testOUt.csv",organized_data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
''' This will put the OG data in full room grid form'''
room_grid_list=[]
for instance in rssiData :
	beacon_list=instance[1]
	act_loc=instance[2]
	# fprint("instance",instance)
	tmp_room_grid=copy.deepcopy(room_grid)
	for i in range(len(beacon_list)) :
		rssi=beacon_list[i]
		if rssi!=-200 :
			loc=beacon_locations[i]
			x_pos=loc[0]
			y_pos=loc[1]
			# print("x={0},y={1}, rssi={2}".format(x_pos,y_pos, rssi))
			tmp_room_grid[y_pos-1][x_pos-1]=rssi #First is Y because of the way it be
	room_grid_list.append([tmp_room_grid, act_loc])

fprint("room_grid",len(room_grid_list))
writeCSV("room_formated.csv",room_grid_list)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This is going to plot out rssi vs dist in og data to determine if the flattening process messed up the data
# lprint("group_list",group_list)

x=[]
y=[]

# # #Pre Flattened
# # for group in group_list :
# # 	for cluster in group :
# # 		for entry in cluster :
# #			# fprint("entry",entry)
# # 			for i in range(len(entry[1])) :
# # 				rssi=entry[1][i]
# # 				if (rssi!=-200) :
# # 					x.append(entry[1][i]) #appends rssi
# # 					y.append(getEucDist(beacon_locations[i],entry[2]))
#
# #Post Flatenned
# for group in flattenedList :
# 	for entry in group :
# 		# fprint("entry",entry)
# 		for i in range(len(entry[1])) :
# 			rssi=entry[1][i]
# 			if (rssi!=-200) :
# 				x.append(entry[1][i]) #appends rssi
# 				y.append(getEucDist(beacon_locations[i],entry[2]))
#
# ##OG DATA
# # for entry in rssiData :
# # 	for i in range(len(entry[1])) :
# # 		rssi=entry[1][i]
# # 		if (rssi!=-200) :
# # 			x.append(entry[1][i]) #appends rssi
# # 			y.append(getEucDist(beacon_locations[i],entry[2]))
#
#
# # print(x)
# plt.plot(y, x, '.')
# plt.title('rssi vs dist of post-edited data, groups of 8(v2)')
# plt.xlabel("Distance (relative)")
# plt.ylabel("rssi (dbm)")
# plt.show()

'''
'''


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# prints out how many datapoints are in the largest chunk
# fprint("biggest ", biggestChunk(closeRssi))

# ScatterPlots all chunks of size greater than a value
# scatterPlot(sizeLimit(chunkByDist(rssiData,12),3))


# chunkedData=chunkByDist(rssiData,12)
# for data in chunkedData :
# 	print("gap")
# 	for variableName in data :
		# fprint("variableName",variableName)




#Notes:
'''
1)
I first realized that there are only 13 datapoints that have more than 3 beacons of data
This is an issue as a minimum of 3 is reaquired for 2D trilateration
2)
I decided to try grouping the datasets by time. e.g. if a timestamp was within n seconds of another timestamp, it follows that they were collected near eachother.

3)
After collecting and viewing this data, I also decided to group them by euclidean distance, which makes a lot more sense.
To do this I assume that each grid point is a unit of 1.

This unfortunately did not result in as many chunks that reach a cutoff as I would have hoped.


4)
At this point the evaluation thingy works, which suggests 5 is an optimal value.
Now I need to figure out how to generate maximum sections within those groups which each have 3 rssi points.

I have made a function that successfully takes in groups of euclidean grouped instances and clusters them into ones with a specifc varience.
This can then be flattened into a single beacon instance that has 3 points

1373 instances for full dataset

This was successful

5)
now I will make the data actually useful.
First I will assign locations to the signal strengths


I guess now I need to prepare the various ways to train.
specifically:
with signal strength
with distance
other options
regression vs classification


TODO:
check if euc distance update fucks shit up

'''
