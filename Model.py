#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:30:20 2021

@author: vindyaliyanage
"""
import pandas as pd
import csv
import re
import random

random.seed(10)

data1 = pd.read_csv("Elephant.csv")
keywords = pd.read_csv("keywords_Elephant.csv")
p = data1.DTBin
print(data1.shape)
#type(p)
#print(data1.head())
#Subset data
data2 = data1[["Caption_Time2", "OpenBin" ,"JudgesBin"]]
#print(data2.shape)
print(keywords)
words = keywords.Labels
#words2 = keywords.Web entities
print(keywords)

#select 1 row to test 
row1 = data2.loc[0,:]
print(len(row1.Caption_Time2))
#print(len(row1))
#print(row1)

# ------------- get synonims

from nltk.corpus import wordnet
import nltk
from collections import OrderedDict 
#nltk.download('wordnet')
syns = wordnet.synsets("Dog")
#print(syns)
#if maching keywords can be generated, Then a keyword count can be used as an attribute


from nltk.corpus import wordnet
synonyms = []
antonyms = []
for w in words:    
    for syn in wordnet.synsets(w):
    	for l in syn.lemmas():
    		synonyms.append(l.name())
    		if l.antonyms():
    			antonyms.append(l.antonyms()[0].name())
    

#the list with all synonyms
print(words)
#print(synonyms)
#print(antonyms)
#append with keywords

my_synonym_list = OrderedDict.fromkeys(synonyms)
my_antonyms_list = OrderedDict.fromkeys(antonyms)
print("----synonyms----")
print(list(my_synonym_list))
print("----antonyms----")
print(list(my_antonyms_list))
keys = []
keys_antonyms = []
keys = list(my_synonym_list)
keys_antonyms = list(my_antonyms_list)


#print(set(antonyms))


#-------------------- Analyse caption ------------------
#Clength = pd.Series([])
Clength = []
KeywordMatch = []
antonymMatch =[]
specialChar = []
for x in range(len(data2)):
    value1 = 0
    value2 = 0
    clength = len(data2.loc[x,:].Caption_Time2) 
    #get length of the string 
    length = len(data2.loc[x,:].Caption_Time2)
    Clength.append(length)
    #Clength = Clength.set_value(r, length)
    
    specialChars = len(re.sub('[^\^&*$!,._-]+' ,'', data2.loc[x,:].Caption_Time2))
    print(data2.loc[x,:].Caption_Time2)
    print("special",specialChars)
    value1 = specialChars/clength
    value2 = round(value2, 4)
    specialChar.append(value2)
    print("Value2------")
    print(value2)
    #remove special characters
    clean = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,']", "",data2.loc[x,:].Caption_Time2)
    #print(clean)
    #remove stopwords and remove capital simple
    
    #split words from the string
    result1 = clean.lstrip()
    #Split string in to words
    result2 = clean.split()
    #print("result2",result2)
    count = 0
    count_antonym = 0 
    for i in result2:
        for j in keys:
            if i.lower() == j.lower():
                count = count + 1
                #show matching keywords
                #print("keyword match", count)
    #add count to keyword array            
    KeywordMatch.append(count/clength)
   # print("Match",count)
    for i in result2:
         for k in keys_antonyms:
            if i.lower() == k.lower():
                count_antonym = count_antonym + 1
                #show matching keywords
                #print("keyword match", count)
    #add count to keyword array            
    antonymMatch.append(count_antonym/clength)

#antonymMatch
#print("antonymMatch")
#print(antonymMatch)
print("specialChar-------")
print(specialChar)

#print("Caption_length", Clength)
data2.insert(3, "Keywords", KeywordMatch)
#add caption length to dataset2 
data2.insert(3, "Caption_length", Clength)
data2.insert(3, "Special_chars", specialChar)
data2.insert(3, "Antonyms", antonymMatch)
print(data2)
#row2 = data2.loc[1,:]
#print(row2)
datax = data2.dropna(axis='rows')
#print("datax",datax)            

data2.to_csv('datafile.csv', sep='\t', index=False)           
#----------------- random forest ----------------
            
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
x = datax[["Caption_length", "OpenBin","Keywords","Special_chars","Antonyms"]] # Features
y = datax.JudgesBin   # Labels

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=200)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#predict test set
#print(x_test)
y_pred=clf.predict(x_test)
#print(y_pred)
x_test.insert(3, "Predicted_level", y_pred)
x_test.insert(2, "Actual_level", y_test)
x_test.to_csv('Test.csv', sep='\t', index=False)  

#------ Confusion matrix
from sklearn.metrics import confusion_matrix
y_true = y_test
y_pred = y_pred
confusion_matrix(y_true, y_pred)

# tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
fig = plt.figure(figsize=(15, 10))

plot_tree(clf.estimators_[5], 
          feature_names=x,
          class_names=y, 
          filled=True, impurity=True, 
          rounded=True)


