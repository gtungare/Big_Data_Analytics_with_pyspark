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

#wget.download('https://metcs777.s3.amazonaws.com/flights-small.csv',out=path)

getdata= path+"flights-small.csv"

flightdata = sqlContext.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(getdata)


print(flightdata.show())

#df=flightdata.select(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])


## using join the solve the question

#dfn1=df.where(((df['ORIGIN_AIRPORT'] == 'LAX' )))
#dfn2=df.where(((df['DESTINATION_AIRPORT'] == 'JFK')))

#one_Stop_flight = dfn1.join(dfn2, "DESTINATION_AIRPORT" , 'inner').toDF("DESTINATION_AIRPORT","ORIGIN_AIRPORT","STOP_AIRPORT")

#print(one_Stop_flight.show())

# Free the (nt)ontext object
sc.stop()

print("end")

