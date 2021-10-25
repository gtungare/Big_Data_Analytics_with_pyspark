"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1 - Exam
Date: Oct 2021

 C:/Users/gaura/PycharmProjects/pythonProject/project-2018-BRFSS-arthritis.csv

"""

from __future__ import print_function
import sys
import sys
from operator import add

from pyspark import SparkContext
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
import sys

from operator import add

from pyspark.sql.types import *
from pyspark.sql import functions as func

from pyspark.sql.functions import lit
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.functions import array
from pyspark import sql, SparkConf, SparkContext

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

# Print the script name
print("PySpark Script: ", sys.argv[0])
sc = SparkContext(appName="Module3_Task1")

# Create a spark context and print some information about the context object

conf = SparkConf().setAppName("Process_mod3_q1")
sqlContext = sql.SQLContext(sc)

import wget

path="C:/Users/gaura/PycharmProjects/pythonProject/exam/"

wget.download('https://metcs777.s3.amazonaws.com/flights-small.csv',out=path)

getdata= path+"flights-small.csv"

flightdata = sqlContext.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(getdata)

df=flightdata.select(['DAY_OF_WEEK','AIRLINE','DEPARTURE_DELAY'])

#print(df.show())

df.withColumn("COUNT", lit(1)).groupBy("DAY_OF_WEEK", "AIRLINE").agg(func.sum("COUNT")).orderBy("sum(COUNT)", ascending=False).limit(5).show()


# Free the (nt)ontext object
sc.stop()

print("end")

