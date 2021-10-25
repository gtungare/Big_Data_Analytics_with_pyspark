"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Question 4.2 Task 2 - Top-10 Best Drivers (15 Points)

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
from pyspark.sql.functions import round

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

# Print the script name
print("PySpark Script: ", sys.argv[0])
sc = SparkContext(appName="PythonWordCount")

# Create a spark context and print some information about the context object

conf = SparkConf().setAppName("Process_q2")

sqlContext = sql.SQLContext(sc)
df1 = sqlContext.read.csv(sys.argv[1])

df=df1.select(['_c1','_c4','_c16']).toDF("hack_license","trip_time_in_secs","total_amount")

dfn=df.withColumn('money_per_minute',round(df['total_amount']/(df['trip_time_in_secs']/60),3))

top_10_best_driver=dfn.where(dfn['money_per_minute'] > 300).orderBy(dfn['money_per_minute'].desc()).head(10)

# now we want to store the result in a single file on the cluster
dataToASingleFile=sc.parallelize(top_10_best_driver).coalesce(1)

dataToASingleFile.saveAsTextFile(sys.argv[1])

#print(top_10_best_driver)

# Free the (nt)ontext object
sc.stop()
print("end")

