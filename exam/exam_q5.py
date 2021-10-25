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
df = sqlContext.read.csv('C:/Users/gaura/PycharmProjects/pythonProject/exam/ex2019-trainingData.csv',inferSchema=True,header=True)


#trainingdata = sqlContext.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(training)

df.show()


sc.stop()

print("end")
