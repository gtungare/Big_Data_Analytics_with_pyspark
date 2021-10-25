"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Question 4.1 Top-10 Active Taxis (5 points)

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

from pyspark import sql, SparkConf, SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

# Print the script name
print("PySpark Script: ", sys.argv[0])
sc = SparkContext(appName="PythonWordCount")

# Create a spark context and print some information about the context object

conf = SparkConf().setAppName("Process_q1")
sqlContext = sql.SQLContext(sc)
df1 = sqlContext.read.csv(sys.argv[1])


## selecting the columns needed
df=df1.select(['_c0','_c1']).toDF("medallion","hack_license")

group_count=df.groupby('medallion').count().toDF('medallion','driver_count')

top_10_active_taxis=group_count.where(group_count['driver_count']>500).orderBy('driver_count').tail(10)

# now we want to store the result in a single file on the cluster for top_10_active_taxis
dataToASingleFile=sc.parallelize(top_10_active_taxis).coalesce(1)

dataToASingleFile.saveAsTextFile(sys.argv[2])

#print(top_10_active_taxis)

# Free the (nt)ontext object
sc.stop()
print("end")

