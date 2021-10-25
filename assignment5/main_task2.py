"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Assignemnt5- Task2 SVN with implementation with RDD

"""

from __future__ import print_function
import sys
import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from operator import add

from  pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.conf import SparkConf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pyspark import sql, SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, date_trunc

from pyspark.sql.functions import date_format
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import hour
from pyspark.sql.functions import minute
from pyspark.sql.functions import second
from pyspark.sql.functions import when
from pyspark.sql.functions import to_date
from pyspark.sql.functions import array

import numpy as np

import re
import sys
import numpy as np
from operator import add
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import time

#########################  Declariing feature size

feature_size=20000

##########################

def freqArray (listOfIndices, numberofwords):
	returnVal = np.zeros (feature_size)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	returnVal = np.divide(returnVal, numberofwords)
	return returnVal

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

sc = SparkContext(appName="Module5_Task1logreg")

d_corpus = sc.textFile(sys.argv[1], 1)

start_read_data = time.time()

d_corpus.cache()
#numberOfDocs = d_corpus.count()

d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')
d_keyAndText.cache()

d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

d_keyAndListOfWords.cache()
########################################################################################################################


# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = d_keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x, 1))
print("Get the top 20,000 words")

# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(add)

# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words.
topWords = allCounts.top(feature_size, lambda x: x[1])

topWordsK = sc.parallelize(range(feature_size))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

dictionary.cache()

def buildArray(listOfIndices):
    returnVal = np.zeros(feature_size)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(feature_size)

    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1

    return returnVal
################### Task 2  ##################

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...

allWordsWithDocID = d_keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allWordsWithDocID.cache()

# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = dictionary.join(allWordsWithDocID)
allDictionaryWords.cache()

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x:(x[1][1],x[1][0]))
justDocAndPos.cache()

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
allDictionaryWordsInEachDoc.cache()
# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
allDocsAsNumpyArrays.cache()

myRDD=allDocsAsNumpyArrays.map(lambda x:[( 1 if ( x[0].startswith('AU')) else 0),x[1]])
myRDD.cache()

from pyspark.sql import SparkSession


d_corpus_test_data = sc.textFile(sys.argv[2], 1)
d_corpus_test_data.cache()

#numberOfDocs = d_corpus.count()

d_keyAndText_test_data = d_corpus_test_data.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')
d_keyAndText_test_data.cache()

d_keyAndListOfWords_test_data = d_keyAndText_test_data.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

d_keyAndListOfWords_test_data.cache()
########################################################################################################################
# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWordsWithDocID_test_data = d_keyAndListOfWords_test_data.flatMap(lambda x: ((j, x[0]) for j in x[1]))
allWordsWithDocID_test_data.cache()


# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords_test_data = dictionary.join(allWordsWithDocID_test_data)
allDictionaryWords_test_data.cache()


# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos_test_data = allDictionaryWords_test_data.map(lambda x:(x[1][1],x[1][0]))
justDocAndPos_test_data.cache()

# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc_test_data = justDocAndPos_test_data.groupByKey()
allDictionaryWordsInEachDoc_test_data.cache()
# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArrays_test_data= allDictionaryWordsInEachDoc_test_data.map(lambda x: (x[0], buildArray(x[1])))
allDocsAsNumpyArrays_test_data.cache()

## Building the new RDD using Test Data set
myRDD_test_data=allDocsAsNumpyArrays_test_data.map(lambda x:[( 1 if ( x[0].startswith('AU')) else 0),x[1]])
myRDD_test_data.cache()

######################################################################################################################

# Load and parse the data
def parsePoint(line):
    #values = [float(x) for x in line.split(' ')]
    return LabeledPoint(line[0], line[1:])

#parsedData = myRDD.map(parsePoint)
my_data = myRDD.map(lambda x: LabeledPoint(x[0], list(x[1])))
my_data_test = myRDD_test_data.map(lambda x: LabeledPoint(x[0], list(x[1])))

end_read_data = time.time()

print (" -------SVN total Read time in sec : ",end_read_data-start_read_data)

## Model training

start_model_training = time.time()

model = SVMWithSGD.train(my_data,regParam=0.2,regType='l2',initialWeights=None)

end_model_training = time.time()

# Evaluating the model on training data

start_model_evaluation = time.time()

labelsAndPreds = my_data_test.map(lambda p: (p.label, model.predict(p.features)))

labelsAndPreds.cache()

end_model_evaluation = time.time()

print ("----------- SVN total Model testing and training time in sec : ",end_model_evaluation-start_model_training)
### lp[0]= Actual
### lp[1]= Prediected
start_conf_matrix = time.time()

tp = labelsAndPreds.filter(lambda lp: ((lp[0]==1.0) and (lp[1] == 1))).count()
print("tp=",tp)
fp = labelsAndPreds.filter(lambda lp: ((lp[0]==0) and (lp[1] == 1))).count()
print("fp=",fp)
fn = labelsAndPreds.filter(lambda lp: ((lp[0]==1) and (lp[1] == 0))).count()
print("fn=",fn)
tn = labelsAndPreds.filter(lambda lp: ((lp[0]==0) and (lp[1] == 0))).count()
print("tn=",tn)
# Calculate F1 Score
f1=2*tp/((2*tp)+fp+fn)

end_conf_matrix = time.time()

print ("----------- Confusion matrix SVN calcuation time in sec : ",end_conf_matrix-start_conf_matrix)

print(" -----SVN  F1 Score----")
print("F1 :",f1)


total_end_time = time.time()

print (" ---- total time SVN in sec :",total_end_time-start_read_data)

# Free the (nt)ontext object
sc.stop()

print("end")
