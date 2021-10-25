"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Task 3 - Implementtion using dataframe

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
sc = SparkContext(appName="Module3_Task3")

# Create a spark context and print some information about the context object

conf = SparkConf().setAppName("Process_mod3_q3")
sqlContext = sql.SQLContext(sc)
df1 = sqlContext.read.csv(sys.argv[1])
df1.cache()
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

# Filtering the record per of taking total amount between 1 and 600
#
# selecting the columns for grouping by
#
dfn=df1.select(['_c1',(col('_c4')/3600),'_c5','_c15',date_format('_c2', 'HH:mm:ss'),'_c16',date_format('_c2',"MM/dd/yyyy")]).where((df1['_c16'] < 600.00 ) & (df1['_c16'] > 1.00 ) ).toDF("driver_id","trip_time","trip_distance","tolls_amount","pickup_time","total_amount","trip_day")
# Implementing logic for night ride
dfn=dfn.withColumn("night_rides_hours", hour(dfn["pickup_time"])).withColumn("night_rides_min", minute(dfn["pickup_time"]))
dfn=dfn.withColumn("flag_night_rides", when( (((dfn["night_rides_hours"] >=1) &( dfn["night_rides_hours"] <=5) & ((dfn["night_rides_min"] >=1) &( dfn["night_rides_min"] <=59) )) ),'1').otherwise(0))

dfn.cache()
## Group by driver and counting for trip for each day
dfnagg=dfn.groupby("driver_id","trip_day").agg({'trip_distance': 'sum','trip_time': 'sum','tolls_amount':'sum','total_amount':'sum','driver_id':'count','flag_night_rides':'sum'}).toDF("driver_id","trip_day","total_trip_distance","total_num_rides","total_night_rides","total_tolls_amount","total_trip_time","total_amount")
dfnagg.cache()

n=dfnagg.count()

print("count",n)
# # Let's start with main iterative part of gradient descent algorithm

## Initalize the learning rate
learningRate =  0.00000001 #graph on this one
#learningRate =   0.00000000002
#learningRate =  0.000001 not good
#learningRate = 0.00001 not good
#learningRate = 0.0001 not good
#earningRate = 0.001
#learningRate = 0.01
num_iteration = 100
m_current=np.zeros(5)
beta = np.zeros(5)
print(m_current)

dfnagg.cache()
# Let's start with main iterative part of gradient descent algorithm
for i in range(num_iteration):
    # Calculate the prediction with current regression coefficients.
    # "total_trip_distance","total_num_rides","total_night_rides","total_tolls_amount","total_trip_time","total_amount")
    #dfnagg = dfnagg.withColumn("y_prediction", np.dot(dfnagg["total_trip_distance","total_num_rides","total_night_rides","total_tolls_amount","total_trip_time"] , m_current))
    dfnagg = dfnagg.withColumn("y_prediction", ((dfnagg["total_trip_distance"]*m_current[0])+(dfnagg["total_num_rides"]*m_current[1])+(dfnagg["total_night_rides"]*m_current[2])+(dfnagg["total_tolls_amount"]*m_current[3])+dfnagg["total_trip_time"]*m_current[1]))
    dfnagg = dfnagg.withColumn("y_ypred", dfnagg["total_amount"] - dfnagg["y_prediction"])
    dfnagg = dfnagg.withColumn("squaare_y_ypred", dfnagg["y_ypred"] * dfnagg["y_ypred"])
    dfnagg = dfnagg.withColumn("x_mult_y_ypred", dfnagg["total_trip_time"] * dfnagg["y_ypred"])
    # We compute costs just for monitoring
    #cost = sum((y - y_prediction) ** 2)
    cost_gd = (dfnagg.agg({'squaare_y_ypred': 'sum'}).collect()[0][0])
    # calculate gradients.
    m_sum = (dfnagg.agg({'x_mult_y_ypred': 'sum'}).collect()[0][0])
    m_gradient = (-1.0 / n) * m_sum
    #print(m_gradient)
    # update the weights - Regression Coefficients
    m_current = m_current - learningRate * m_gradient
    print(i, "m_gradient=", m_gradient, " Cost=", cost_gd,"m_current",m_current)


# Free the (nt)ontext object
sc.stop()

print("end")

