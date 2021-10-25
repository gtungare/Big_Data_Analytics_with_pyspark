"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1 - Exam
Date: Oct 2021

NOte

Argument 1 - C:/Users/gaura/PycharmProjects/pythonProject/exam/ex2019-trainingData.csv
Argument 2 - C:/Users/gaura/PycharmProjects/pythonProject/exam/ex2019-testingData.csv

"""

from __future__ import print_function
import sys
import os

import pylab as pl
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
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

##########################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

sc = SparkContext(appName="CS777_Project")

from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Process_q1")
sqlContext = sql.SQLContext(sc)
df_train = sqlContext.read.csv(sys.argv[1],inferSchema=True,header=True)

df_test = sqlContext.read.csv(sys.argv[2],inferSchema=True,header=True)


import matplotlib.pyplot as plt


x=df_test.select(['f1']).collect()
y=df_test.select(['f2']).collect()
# Set the figure size

# Scatterplot
plt.scatter(x,y)
pl.title("Test Data")
plt.xlabel("f1")
plt.ylabel("f2")
plt.show()

x=df_train.select(['f1']).collect()
y=df_train.select(['f2']).collect()
# Set the figure size
#plt.figure(figsize=(10, 10))
# Scatterplot
plt.scatter(x,y)
pl.title("Training")
plt.xlabel("f1")
plt.ylabel("f2")
plt.show()

## Reducing the trainign set

df_train=df_train.where((df_train['f2'] >= 20) |( df_train['f2'] <= 20))

from pyspark.sql.functions import when

#dfn=df.withColumn('class', when(df['havarth3']==0,0).otherwise(1) )



from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)


assembler = VectorAssembler(inputCols=['f1', 'f2'], outputCol='features')

log_reg = LinearSVC(featuresCol='features',labelCol='label',maxIter=100)

#train_data, test_data = df_train.randomSplit([0.66,0.33])

#df_test = sqlContext.read.csv(sys.argv[2],inferSchema=True,header=True)

pipeline = Pipeline(stages=[assembler,log_reg])

fit_model = pipeline.fit(df_train)

results = fit_model.transform(df_test)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='label')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Get metrics

weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})

print("SVM - weightedPrecision ",weightedPrecision)
print("SVM - weightedRecall",weightedRecall)

#####################################################################################################################

from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol='features',labelCol='label',maxIter=100)

#train_data, test_data = df_train.randomSplit([0.66,0.33])

#df_test = sqlContext.read.csv(sys.argv[2],inferSchema=True,header=True)

pipeline = Pipeline(stages=[assembler,log_reg])

fit_model = pipeline.fit(df_train)

results = fit_model.transform(df_test)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='label')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Get metrics
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})

print("LogisticRegression - weightedPrecision ",weightedPrecision)
print("LogisticRegression - weightedRecall",weightedRecall)

sc.stop()

print("end")
