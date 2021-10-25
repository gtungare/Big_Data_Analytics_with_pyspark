"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Task 2 - Module 3 -Task 2

Implementation is done using dataframe


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

import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

# Print the script name
print("PySpark Script: ", sys.argv[0])
sc = SparkContext(appName="Module3_Task2")

# Create a spark context and print some information about the context object

conf = SparkConf().setAppName("Process_mod3_q2")
sqlContext = sql.SQLContext(sc)
df1 = sqlContext.read.csv(sys.argv[1])

from pyspark.sql import SparkSession
from pyspark.sql.types import *


# Filtering the record per of taking total amount between 1 and 600


dfn=df1.select(['_c5','_c11']).where((df1['_c16'] < 600.00 ) & (df1['_c16'] > 1.00 ) ).toDF("trip_distance","total_amount")

n=dfn.count()
print("count",n)

## Initalize the learning rate
learningRate = 0.000000001
num_iteration = 100
m_current=0.1

beta = np.zeros(1)
#print(beta)

dfn.cache()
# Let's start with main iterative part of gradient descent algorithm
for i in range(num_iteration):
    # Calculate the prediction with current regression coefficients.

    dfn = dfn.withColumn("y_prediction", dfn["trip_distance"] * m_current)
    dfn = dfn.withColumn("y_ypred", dfn["total_amount"] - dfn["y_prediction"])
    dfn = dfn.withColumn("squaare_y_ypred", dfn["y_ypred"] * dfn["y_ypred"])
    dfn = dfn.withColumn("x_mult_y_ypred", dfn["trip_distance"] * dfn["y_ypred"])
    # We compute costs just for monitoring
    #cost = sum((y - y_prediction) ** 2)
    cost_gd = (dfn.agg({'squaare_y_ypred': 'sum'}).collect()[0][0])
    # calculate gradients.
    m_sum = (dfn.agg({'x_mult_y_ypred': 'sum'}).collect()[0][0])
    m_gradient = (-1.0 / n) * m_sum
    #print(m_gradient)
    # update the weights - Regression Coefficients
    m_current = m_current - learningRate * m_gradient
    print(i, "m_gradient=", m_gradient, " Cost=", cost_gd,"m_current",m_current)

    
# Free the (nt)ontext object
sc.stop()

print("end")

