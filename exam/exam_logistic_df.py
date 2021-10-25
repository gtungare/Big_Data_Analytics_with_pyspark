"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1 - Exam
Date: Oct 2021


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
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

##########################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

sc = SparkContext(appName="CS777_Project")

from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Process_q1")
sqlContext = sql.SQLContext(sc)
#df = sqlContext.read.csv(sys.argv[1],inferSchema=True,header=True)


import wget

path="C:/Users/gaura/PycharmProjects/pythonProject/exam/"

#wget.download('https://metcs777.s3.amazonaws.com/ex2019-trainingData.csv',out=path)
#wget.download('https://metcs777.s3.amazonaws.com/ex2019-testingData.csv ',out=path)

training= path+"ex2019-trainingData.csv"
testset= path+"ex2019-trainingData.csv"

path2="C:/Users/gaura/PycharmProjects/pythonProject/exam/ex2019-trainingData.csv"

df1 = sqlContext.read.csv(path2)

#trainingdata = sqlContext.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(training)

df1.show()

exit()

df1=df1.select(['employ1', 'income2','havarth3'])

from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)


assembler = VectorAssembler(inputCols=['employ1', 'income2'], outputCol='features')

log_reg = LogisticRegression(featuresCol='features',labelCol='havarth3',maxIter=100)

train_data, test_data = df1.randomSplit([0.66,0.33])

pipeline = Pipeline(stages=[assembler,log_reg])

fit_model = pipeline.fit(train_data)

results = fit_model.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

sc.stop()

print("end")
