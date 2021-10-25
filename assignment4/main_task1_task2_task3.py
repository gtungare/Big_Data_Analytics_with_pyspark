"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Assignemnt4- Task 1, Task2 and Task 3  with implementation with RDD

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

def freqArray (listOfIndices, numberofwords):
	returnVal = np.zeros (20000)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	returnVal = np.divide(returnVal, numberofwords)
	return returnVal

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

sc = SparkContext(appName="Module4_Task1")

d_corpus = sc.textFile(sys.argv[1], 1)

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
topWords = allCounts.top(20000, lambda x: x[1])

topWordsK = sc.parallelize(range(20000))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

dictionary.cache()

print("-----Print Task 1---------")

print("Task1 :",dictionary.filter(lambda  x:"applicant" in x).collect())
print("Task1 :",dictionary.filter(lambda  x:"and" in x).collect())
print("Task1 :",dictionary.filter(lambda  x:"attack" in x).collect())
print("Task1 :",dictionary.filter(lambda  x:"protein" in x).collect())
print("Task1 :",dictionary.filter(lambda  x:"car" in x).collect())

#############################################################################################

print("------Starting Task 2 ------")

def buildArray(listOfIndices):
    returnVal = np.zeros(20000)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(20000)

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

reg_lambda=0.00001
#print(myRDD.take(5))
learningRate = 0.0000001
num_iteration = 300
#size= n

beta=np.zeros(20000)

for i in range(num_iteration):

    Cost_LLH1 = myRDD.map(lambda x: ((x[0], ( (np.dot(x[1], beta)*x[0])+ (  np.log(1+ ( np.exp(np.dot(x[1], beta))    )  ) ) + (reg_lambda*(beta**2))           )))).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    gradient = myRDD.map(lambda x: ((x[0], ( (np.dot(x[1], x[0]))+ (np.dot(x[1],  (( np.exp(np.dot(x[1], beta)))/(1+ ( np.exp(np.dot(x[1], beta))))))) + (2*reg_lambda*beta)    )))).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    cost = Cost_LLH1[1]
    print( i,"Beta", beta, " Cost", cost[1])
    beta = beta - learningRate * gradient[1]


#printing  out the  five words with the largest regression coefficients

K = 5

#  Collecting the largest coefficient
ind=np.argpartition(beta,-K)[-K:]

print("--Task2 -> printing  out the  five words with the largest regression coefficients-------")

print("Task2 :",dictionary.filter(lambda  x: ind[0] in x).collect())
print("Task2 :",dictionary.filter(lambda  x: ind[1] in x).collect())
print("Task2 :",dictionary.filter(lambda  x: ind[2] in x).collect())
print("Task2 :",dictionary.filter(lambda  x: ind[3] in x).collect())
print("Task2 :",dictionary.filter(lambda  x: ind[4] in x).collect())

from pyspark.sql import SparkSession


##################################### Task3  ###################################################################################

print("----Starting Task 3---- ")

#test_data_set='C:/Users/gaura/PycharmProjects/pythonProject/TestingData.txt'
#test_data_set='gs://metcs777/TestingData.txt'

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


## Calculte the theta for test data set-- 0 if θ < 0 , and 1 if θ > 0

#theta = myRDD_test_data.map(lambda x: (x[0], (np.dot(x[1], beta)),1 if (np.dot(x[1], beta) > 0 ) else 0))
theta = myRDD_test_data.map(lambda x: (x[0],1 if (np.dot(x[1], beta) > 0 ) else 0))
theta.cache()

sqlContext = sql.SQLContext(sc)
columns = StructType([StructField('y_act',IntegerType(), False),
                     StructField('ypred',IntegerType(), False)])
dfn = sqlContext.createDataFrame(data=theta,schema=columns)
dfn.cache()

## Calaculating the confusion matrix

# Calculate True Positive
tp=dfn.where(((dfn['y_act'] == 1 ) & (dfn['ypred']==1))).count()
# Calculate False Positive
fp=dfn.where(((dfn['y_act'] == 0 ) & (dfn['ypred']==1))).count()
# Calculate False Negative
fn=dfn.where(((dfn['y_act'] == 1 ) & (dfn['ypred']==0))).count()
# Calculate True Negative
tn=dfn.where(((dfn['y_act'] == 0 ) & (dfn['ypred']==0))).count()
# Calculate F1 Score
f1=2*tp/((2*tp)+fp+fn)
print(" ---Task 3 ->----F1 Score----")
print("F1 :",f1)

########################################################################################################################

#
# Free the (nt)ontext object
sc.stop()

print("end")
