"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Project Implementation

1. Classificaion using CDC data

2. CLustering using gapminder

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
df = sqlContext.read.csv(sys.argv[1],inferSchema=True,header=True)


df1=df.select(['employ1', 'income2', 'weight2', 'height3', 'children', 'veteran3', 'blind', 'renthom1',
               'sex1', 'marital', 'educa', 'deaf', 'decide', 'drnkdrv', 'flushot6', 'seatbelt', 'hivtst6', 'hivrisk5',
               'pneuvac4', 'alcday5', 'diffwalk', 'usenow3', 'diffdres', 'diffalon', 'smoke100', 'rmvteth4', 'lastden4',
               'diabete3', 'psu', 'physhlth', 'menthlth', 'hlthpln1', 'genhlth', 'dispcode', 'chckdny1', 'iyear', 'fmonth',
               'imonth', 'iday', 'persdoc2', 'medcost', 'checkup1', 'exerany2', 'chcocncr', 'chccopd1', 'addepev2', 'chcscncr',
               'asthma3', 'cvdstrk3', 'sleptim1', 'cvdinfr4', 'cvdcrhd4', 'qstver', 'qstlang', 'metstat', 'htin4', 'wtkg3',
               'bmi5', 'bmi5cat', 'htm4', 'ageg', 'raceg21', 'age80', 'raceg1', 'ageg5yr', 'age65yr', 'rfbmi5', 'chldcnt',
               'educag', 'incomg', 'rfdrhv6', 'rfseat2', 'rfseat3', 'drnkwek', 'rfbing5', 'smoker3', 'rfsmok3', 'drnkany5',
               'racegr3', 'race', 'urbstat', 'chispnc', 'llcpwt2', 'llcpwt', 'rfhlth', 'dualuse', 'imprace', 'hispanc',
               'wt2rake', 'ststr', 'strwt', 'rawrake', 'phys14d', 'ment14d', 'hcvu651', 'totinda', 'denvst3', 'prace1',
               'mrace1', 'exteth3','asthms1', 'michd', 'ltasth1', 'casthm1', 'state','havarth3'])

from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)


assembler = VectorAssembler(inputCols=['employ1', 'income2', 'weight2', 'height3', 'children', 'veteran3', 'blind', 'renthom1',
               'sex1', 'marital', 'educa', 'deaf', 'decide', 'drnkdrv', 'flushot6', 'seatbelt', 'hivtst6', 'hivrisk5',
               'pneuvac4', 'alcday5', 'diffwalk', 'usenow3', 'diffdres', 'diffalon', 'smoke100', 'rmvteth4', 'lastden4',
               'diabete3', 'psu', 'physhlth', 'menthlth', 'hlthpln1', 'genhlth', 'dispcode', 'chckdny1', 'iyear', 'fmonth',
               'imonth', 'iday', 'persdoc2', 'medcost', 'checkup1', 'exerany2', 'chcocncr', 'chccopd1', 'addepev2', 'chcscncr',
               'asthma3', 'cvdstrk3', 'sleptim1', 'cvdinfr4', 'cvdcrhd4', 'qstver', 'qstlang', 'metstat', 'htin4', 'wtkg3',
               'bmi5', 'bmi5cat', 'htm4', 'ageg', 'raceg21', 'age80', 'raceg1', 'ageg5yr', 'age65yr', 'rfbmi5', 'chldcnt',
               'educag', 'incomg', 'rfdrhv6', 'rfseat2', 'rfseat3', 'drnkwek', 'rfbing5', 'smoker3', 'rfsmok3', 'drnkany5',
               'racegr3', 'race', 'urbstat', 'chispnc', 'llcpwt2', 'llcpwt', 'rfhlth', 'dualuse', 'imprace', 'hispanc',
               'wt2rake', 'ststr', 'strwt', 'rawrake', 'phys14d', 'ment14d', 'hcvu651', 'totinda', 'denvst3', 'prace1',
               'mrace1', 'exteth3','asthms1', 'michd', 'ltasth1', 'casthm1', 'state'], outputCol='features')


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

schema = StructType([ \
    StructField("Type",StringType(),True),
    StructField("K",IntegerType(),False),
    StructField("Accuracy",FloatType(),True),
    StructField("F1Score",FloatType(),True),
    StructField("Precision",FloatType(),True),
    StructField("Recall",FloatType(),True)
  ])
data2=[("Logitic Regression",0,acc,f1,weightedPrecision,weightedRecall)]

df_consolidated = sqlContext.createDataFrame(data=data2,schema=schema)

############################### Decision Tree ####################################################

from pyspark.ml.classification import DecisionTreeClassifier

log_dec_tree = DecisionTreeClassifier(featuresCol='features',labelCol='havarth3')

train_data, test_data = df1.randomSplit([0.66,0.33])

pipeline = Pipeline(stages=[assembler,log_dec_tree])

fit_model_dec_tree = pipeline.fit(train_data)

results = fit_model_dec_tree.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

newRow = sqlContext.createDataFrame([("Decision Tree",0,acc,f1,weightedPrecision,weightedRecall)])

df_consolidated=df_consolidated.union(newRow)

############################### Random Forest Classifier Tree ####################################################

from pyspark.ml.classification import RandomForestClassifier

log_random_Forest = RandomForestClassifier(featuresCol='features',labelCol='havarth3')
train_data, test_data = df1.randomSplit([0.66,0.33])
pipeline = Pipeline(stages=[assembler,log_random_Forest])

fit_model_random_forest = pipeline.fit(train_data)

results = fit_model_random_forest.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

newRow = sqlContext.createDataFrame([("Random Forest",0,acc,f1,weightedPrecision,weightedRecall)])

df_consolidated=df_consolidated.union(newRow)


############################### NaiveBayes ####################################################

from pyspark.ml.classification import NaiveBayes

log_naive_bayes = NaiveBayes(featuresCol='features',labelCol='havarth3')

train_data, test_data = df1.randomSplit([0.66,0.33])

pipeline = Pipeline(stages=[assembler,log_naive_bayes])

fit_model_grd_boosted_tree = pipeline.fit(train_data)

results = fit_model_grd_boosted_tree.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

newRow = sqlContext.createDataFrame([("Naive Bayes ",0,acc,f1,weightedPrecision,weightedRecall)])

df_consolidated=df_consolidated.union(newRow)

print(df_consolidated.show())
############__________________________ With Dimentionality Recduction ___________________######################

print(" -------With Dimentionality Recduction ------------")

df1=df.select(['employ1', 'income2', 'weight2','sex1', 'marital',
               'pneuvac4', 'alcday5', 'diffwalk', 'usenow3', 'diffdres', 'diffalon', 'smoke100', 'rmvteth4', 'lastden4',
               'diabete3', 'chcocncr', 'chccopd1', 'addepev2', 'chcscncr',
               'asthma3', 'cvdstrk3', 'sleptim1', 'cvdinfr4', 'cvdcrhd4', 'qstver', 'qstlang', 'metstat', 'htin4', 'wtkg3',
               'bmi5', 'bmi5cat', 'htm4', 'ageg', 'raceg21', 'age80', 'raceg1', 'ageg5yr', 'age65yr', 'rfbmi5', 'chldcnt',
               'educag', 'incomg', 'smoker3', 'rfsmok3',
               'racegr3', 'race', 'urbstat', 'chispnc', 'llcpwt2', 'llcpwt', 'rfhlth', 'dualuse', 'imprace', 'hispanc',
               'mrace1', 'exteth3','asthms1', 'michd', 'ltasth1', 'casthm1', 'state','havarth3'])

assembler = VectorAssembler(inputCols=['employ1', 'income2', 'weight2','sex1', 'marital',
               'pneuvac4', 'alcday5', 'diffwalk', 'usenow3', 'diffdres', 'diffalon', 'smoke100', 'rmvteth4', 'lastden4',
               'diabete3', 'chcocncr', 'chccopd1', 'addepev2', 'chcscncr',
               'asthma3', 'cvdstrk3', 'sleptim1', 'cvdinfr4', 'cvdcrhd4', 'qstver', 'qstlang', 'metstat', 'htin4', 'wtkg3',
               'bmi5', 'bmi5cat', 'htm4', 'ageg', 'raceg21', 'age80', 'raceg1', 'ageg5yr', 'age65yr', 'rfbmi5', 'chldcnt',
               'educag', 'incomg', 'smoker3', 'rfsmok3',
               'racegr3', 'race', 'urbstat', 'chispnc', 'llcpwt2', 'llcpwt', 'rfhlth', 'dualuse', 'imprace', 'hispanc',
               'mrace1', 'exteth3','asthms1', 'michd', 'ltasth1', 'casthm1', 'state'], outputCol='features')

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

schema = StructType([ \
    StructField("Type",StringType(),True),
    StructField("K",IntegerType(),False),
    StructField("Accuracy",FloatType(),True),
    StructField("F1Score",FloatType(),True),
    StructField("Precision",FloatType(),True),
    StructField("Recall",FloatType(),True)
  ])
data2=[("Logitic Regression",0,acc,f1,weightedPrecision,weightedRecall)]

df_consolidated_reduction = sqlContext.createDataFrame(data=data2,schema=schema)

############################### Decision Tree ####################################################

from pyspark.ml.classification import DecisionTreeClassifier

log_dec_tree = DecisionTreeClassifier(featuresCol='features',labelCol='havarth3')

train_data, test_data = df1.randomSplit([0.66,0.33])

pipeline = Pipeline(stages=[assembler,log_dec_tree])

fit_model_dec_tree = pipeline.fit(train_data)

results = fit_model_dec_tree.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

newRow = sqlContext.createDataFrame([("Decision Tree",0,acc,f1,weightedPrecision,weightedRecall)])

df_consolidated_reduction=df_consolidated_reduction.union(newRow)

############################### Random Forest Classifier Tree ####################################################

from pyspark.ml.classification import RandomForestClassifier

log_random_Forest = RandomForestClassifier(featuresCol='features',labelCol='havarth3')
train_data, test_data = df1.randomSplit([0.66,0.33])
pipeline = Pipeline(stages=[assembler,log_random_Forest])

fit_model_random_forest = pipeline.fit(train_data)

results = fit_model_random_forest.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

newRow = sqlContext.createDataFrame([("Random Forest",0,acc,f1,weightedPrecision,weightedRecall)])

df_consolidated_reduction=df_consolidated_reduction.union(newRow)

############################### NaiveBayes ####################################################

from pyspark.ml.classification import NaiveBayes

log_naive_bayes = NaiveBayes(featuresCol='features',labelCol='havarth3')

train_data, test_data = df1.randomSplit([0.66,0.33])

pipeline = Pipeline(stages=[assembler,log_naive_bayes])

fit_model_grd_boosted_tree = pipeline.fit(train_data)

results = fit_model_grd_boosted_tree.transform(test_data)

my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                       labelCol='havarth3')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="havarth3", predictionCol="prediction")

# Get metrics
acc = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(results, {evaluatorMulti.metricName: "weightedRecall"})
auc = my_eval.evaluate(results)

newRow = sqlContext.createDataFrame([("Naive Bayes ",0,acc,f1,weightedPrecision,weightedRecall)])

df_consolidated_reduction=df_consolidated_reduction.union(newRow)

print(df_consolidated_reduction.show())


#####################################################################################################
#pip install gapminder

from gapminder import gapminder

print("-------------------Moving on to gapminder dataset for clustering --------------------------------------")
print("-------------------Attention - pip install gapminder  --------------------------------------")

print(gapminder.head())


# import matplotlib
import matplotlib.pyplot as plt

# Subset of the data for year 1952
data1952 = gapminder

# Set the figure size
plt.figure(figsize=(10, 10))
# Scatterplot
plt.scatter(
    x = data1952['lifeExp'],
    y = data1952['gdpPercap'],
    #s=data1952['pop']/50000,
    cmap="Accent",
    alpha=0.6,
    edgecolors="white",
    linewidth=2);
# Add titles (main and on axis)
plt.yscale('log')
plt.xlabel("Life Expectancy")
plt.ylabel("GDP - per Capita")
plt.title("GapMinder - Life Expectancy Vs GDP - per Capita ")

dfn1=sqlContext.createDataFrame(gapminder)

dfnn=dfn1.select(['pop', 'gdpPercap','lifeExp'])

########_______________________________ K mean _________________________

print("-------- Kmean clustering----------------")
from pyspark.ml.clustering import KMeans

assembler = VectorAssembler(inputCols=['pop', 'gdpPercap','lifeExp'], outputCol='features')

from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='nlifeExp', featuresCol='features',
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,10,2):

    log_kmean = KMeans(featuresCol='features',predictionCol='nlifeExp',k=i)

    train_data, test_data = dfnn.randomSplit([0.7,0.3])

    pipeline = Pipeline(stages=[assembler,log_kmean])

    fit_model_kmean = pipeline.fit(dfnn)

    results = fit_model_kmean.transform(dfnn)

    #------------

    score=evaluator.evaluate(results)
    silhouette_score.append(score)
    print("Silhouette Score:",score)

#pddf_pred = results.toPandas()

#color=['blue','green','cyan', 'black']
#plt.figure(figsize=(12,10))
#plt.scatter(pddf_pred.lifeExp, pddf_pred.gdpPercap,c=pddf_pred.nlifeExp)
#plt.show()

#Visualizing the silhouette scores in a plot

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
plt.plot(range(2,10,2),silhouette_score)
plt.xlabel("K")
plt.ylabel("Score")
pl.title("Silhouette Coefficient plot-K Vs Cost")
plt.show()

########_______________________________ Gaussian Mixture _________________________

print("-------------Gaussian Mixture Clustering --------------------")

from pyspark.ml.clustering import GaussianMixture

assembler = VectorAssembler(inputCols=['pop', 'gdpPercap','lifeExp'], outputCol='features')

from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='nlifeExp', featuresCol='features',
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

#for i in range(45):

log_gaussian = GaussianMixture(featuresCol='features',predictionCol='nlifeExp',k=5)

train_data, test_data = dfnn.randomSplit([0.66,0.33])

pipeline = Pipeline(stages=[assembler,log_gaussian])

fit_model_gaussian = pipeline.fit(dfnn)

results = fit_model_gaussian.transform(dfnn)

print(results.show())

# Free the (nt)ontext object

sc.stop()

print("end")
