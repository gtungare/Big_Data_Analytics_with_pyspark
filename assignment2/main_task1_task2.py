#!/usr/bin/env python
# coding: utf-8
"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Assignment 2 - Task 1 and  2 added 

Please note - The template given by the prof is used

Executes code for both taask 1 and 2 
"""
# In[1]:


import sys
import re
import numpy as np
from operator import add
from numpy import dot
from numpy.linalg import norm

from pyspark import sql, SparkConf, SparkContext

# In[2]:


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

#sc = SparkContext(appName="mod2_task1_task2", conf=SparkConf().set('spark.driver.memory', '24g').set('spark.executor.memory', '12g'))
sc = SparkContext(appName="mod2_task1_task2")


# Set the file paths on your local machine
# Change this line later on your python script when you want to run this on the CLOUD (GC or AWS)

## local
#wiki_pages_file="C:/Users/gaura/Downloads/CS777/assignment2/WikipediaPagesOneDocPerLine1000LinesSmall.txt.bz2"
#wiki_category_file="C:/Users/gaura/Downloads/CS777/assignment2/wiki-categorylinks-small.csv.bz2"
## Small
#wiki_pages_file="gs://metcs777/WikipediaPagesOneDocPerLine1000LinesSmall.txt"
#wiki_category_file="gs://metcs777/wiki-categorylinks.csv.bz2"
# large
wiki_pages_file="gs://metcs777/WikipediaPagesOneDocPerLine1m.txt"
wiki_category_file="gs://metcs777/wiki-categorylinks.csv.bz2"

# In[3]:

# Read two files into RDDs
wikiCategoryLinks=sc.textFile(wiki_category_file)
wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))
#aa=wikiCats.take(1)
#print(aa)
# Now the wikipages
# Now the wikipages
wiki_pages = sc.textFile(wiki_pages_file)

#wikiCategoryLinks.take(2)


# In[4]:


#wikiCats.take(1)


# In[5]:


from pyspark.sql.session import SparkSession
spark = SparkSession(sc)
#df = spark.read.csv(wikiPagesFile)

# Uncomment this line if you want to take look inside the file.
# df.take(1)


# In[6]:


# wikiPages.take(1)


# In[7]:


# Assumption: Each document is stored in one line of the text file
# We need this count later ...
numberOfDocs = wiki_pages.count()

print(numberOfDocs)
# Each entry in validLines will be a line from the text file
#validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)

#abc=validLines.toDF()
# Now, we transform it into a set of (docID, text) pairs


#wiki_p.take(2)
# Assumption: Each document is stored in one line of the text file
# We need this count later ...
numberOfDocs = wiki_pages.count()
# Each entry in validLines will be a line from the text file
validLines = wiki_pages.filter(lambda x : 'id' in x and 'url=' in x)
# Now, we transform it into a set of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))


# keyAndText.take(1)


# In[8]:


def buildArray(listOfIndices):
    returnVal = np.zeros(5000)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1

    mysum = np.sum(returnVal)

    returnVal = np.divide(returnVal, mysum)

    return returnVal


def build_zero_one_array(listOfIndices):
    returnVal = np.zeros(5000)

    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1

    return returnVal


def stringVector(x):
    returnVal = str(x[0])
    for j in x[1]:
        returnVal += ',' + str(j)
    return returnVal


def cousinSim(x, y):
    normA = np.linalg.norm(x)
    normB = np.linalg.norm(y)
    return np.dot(x, y) / (normA * normB)

# In[9]:

# Now, we transform it into a set of (docID, text) pairs
#keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

# Now, we split the text in each (docID, text) pair into a list of words
# After this step, we have a data set with
# (docID, ["word1", "word2", "word3", ...])
# We use a regular expression here to make
# sure that the program does not break down on some of the documents
# remove all non unnecessary characters
regex = re.compile('[^a-zA-Z]')

# remove all non letter characters
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
# better solution here is to use NLTK tokenizer

# Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: x[1]).map(lambda x: (x, 1))


# Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey(add)

print("Get the top 20,000 words")

# Get the top 20,000 words in a local array in a sorted format based on frequency
# If you want to run it on your laptio, it may a longer time for top 20k words.
topWords = allCounts.top(5000, lambda x: x[1])


# now we want to store the result in a single file on the cluster

top10WordsToASingleFile=allCounts.top(10, key=lambda x: x[1])

print("save top10WordsToASingleFile ")
savetop10WordsToASingleFile=sc.parallelize(top10WordsToASingleFile).coalesce(1)
#savetop10WordsToASingleFile.saveAsTextFile(sys.argv[1])
#
print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

# We'll create a RDD that has a set of (word, dictNum) pairs
# start by creating an RDD that has the number 0 through 5000
# 5000 is the number of words that will be in our dictionary
topWordsK = sc.parallelize(range(5000))

# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
# ("NextMostCommon", 2), ...
# the number will be the spot in the dictionary used to tell us
# where the word is located
dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

#last_20_words_in_20k=dictionary.top(20, lambda x : x[1])

print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))

# storing in a file
#savelast_20_words_in_20kToASingleFile=sc.parallelize(last_20_words_in_20k).coalesce(1)
#savelast_20_words_in_20kToASingleFile.saveAsTextFile(sys.argv[1])

print("Task 1 Done ")

# In[10]:


########################################### TASK 2  ##################

# In[10]:

print("Starting Task 2 ")
################### Task 2  ##################

# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...

allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))


# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = dictionary.join(allWordsWithDocID)

# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map(lambda x:(x[1][1],x[1][0]))


# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()


# The following line this gets us a set of
# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positions to a bag-of-words numpy array...
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

print(allDocsAsNumpyArrays.take(3))


# In[11]:


# Now, create a version of allDocsAsNumpyArrays where, in the array,
# every entry is either zero or one.
# A zero means that the word does not occur,
# and a one means that it does.

zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0],np.where(x[1]!=0,1,0)))

# Now, add up all of those arrays into a single array, where the
# i^th entry tells us how many
# individual documents the i^th word in the dictionary appeared in
dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

# Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
multiplier = np.full(5000, numberOfDocs)

# Get the version of dfArray where the i^th entry is the inverse-document frequency for the
# i^th word in the corpus
idfArray = np.log(np.divide(np.full(5000, numberOfDocs),dfArray))

# Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

allDocsAsNumpyArraysTFidf.take(2)

# use the buildArray function to build the feature array
# allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))


# print(allDocsAsNumpyArraysTFidf.take(2))


# In[12]:


#wikiCats.take(1)


# In[19]:


# Now, we join it with categories, and map it after join so that we have only the wikipageID
# This joun can take time on your laptop.
# You can do the join once and generate a new wikiCats data and store it. Our WikiCategories includes all categories
# of wikipedia.

featuresRDD = wikiCats.join(allDocsAsNumpyArraysTFidf).map(lambda x: (x[1][0], x[1][1]))

# Cache this important data because we need to run kNN on this data set.

#print("featuresRDD.count()",featuresRDD.count())

featuresRDD.cache()


#featuresRDD.take(10)


# In[14]:


# Let us count and see how large is this data set.
#wikiAndCatsJoind.count()


# In[20]:


# Finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
def getPrediction(textInput, k):
    # Create an RDD out of the textIput
    myDoc = sc.parallelize(('', textInput))
    print('1..myDoc')
    # Flat map the text to (word, 1) pair for each word in the doc
    wordsInThatDoc = myDoc.flatMap(lambda x: ((j, 1) for j in regex.sub(' ', x).lower().split()))
    print('2..wordsInThatDoc')
    # This will give us a set of (word, (dictionaryPos, 1)) pairs
    allDictionaryWordsInThatDoc = dictionary.join(wordsInThatDoc).map(lambda x: (x[1][1], x[1][0])).groupByKey()

    print('3...allDictionaryWordsInThatDoc')
    # Get tf array for the input string
    myArray = buildArray(allDictionaryWordsInThatDoc.top(1)[0][1])

    print('4...myArray')
    # Get the tf * idf array for the input string
    myArray = np.multiply(dfArray,idfArray)

    #print("len(myArray)",len(myArray))

    print('5...tf * idf')
    # Get the distance from the input text string to all database documents, using cosine similarity (np.dot() )
    #distances = featuresRDD.map(lambda x: (x[0], np.dot(x[1], myArray)))

    distances = allDocsAsNumpyArraysTFidf.map (lambda x : (x[0], cousinSim (x[1],myArray)))
    print('6...distances')

    # get the top k distances
    topK = distances.top(k, lambda x: x[1])

    print('7...topK')
    # and transform the top k distances into a set of (docID, 1) pairs
    docIDRepresented = sc.parallelize(topK).map(lambda x: (x[0], 1))

    print('8...docIDRepresented')
    # now, for each docID, get the count of the number of times this document ID appeared in the top k
    numTimes = docIDRepresented.reduceByKey(lambda x, y: x+y)

    print('9...numTimes')
    # Return the top 1 of them.
    # Ask yourself: Why we are using twice top() operation here?
    print('10...return')
    return numTimes.top(k, lambda x: x[1])

# In[21]:

print("getPrediction_1")
#getPrediction_1 =getPrediction('Sport Basketball Volleyball Soccer', 10)

print(getPrediction('Sport Basketball Volleyball Soccer', 100))

##Storing in a file
#datagetPrediction_1ToASingleFile=sc.parallelize(getPrediction_1).coalesce(1)
#datagetPrediction_1ToASingleFile.saveAsTextFile(sys.argv[1])

# In[22]:
print("getPrediction_2")

#getPrediction_2 =getPrediction('What is the capital city of Australia?', 10)
print(getPrediction('What is the capital city of Australia?', 100))

##Storing in a file
#datagetPrediction_2ToASingleFile=sc.parallelize(getPrediction_2).coalesce(1)
#datagetPrediction_2ToASingleFile.saveAsTextFile(sys.argv[1])

# In[23]:

print("getPrediction_3")
#getPrediction_3=getPrediction('How many goals Vancouver score last year?', 10)

print(getPrediction('How many goals Vancouver score last year?', 100))
##Storing in a file
#datagetPrediction_3ToASingleFile=sc.parallelize(getPrediction_3).coalesce(1)
#datagetPrediction_3ToASingleFile.saveAsTextFile(sys.argv[1])
