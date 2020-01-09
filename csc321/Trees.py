#!/usr/bin/env python
# coding: utf-8

# # CSC-321: Data Mining and Machine Learning
# 
# ## Working with scikit-learn - Trees
# 
# In this notebook, I'll demonstrate a decision tree, using the IRIS data set
# It uses another library - graphviz - to display the final tree
# 

# In[8]:


import graphviz 
from sklearn import tree
from sklearn.datasets import load_iris

#print("Done")

iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

