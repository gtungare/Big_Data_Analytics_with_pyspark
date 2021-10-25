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

df=flightdata.select(['DAY_OF_WEEK','DEPARTURE_DELAY','CANCELLED'])

week1=df.where(df['DAY_OF_WEEK'] == 1)#.head(3)
total_flight_Week1=week1.count()
sum_cancellation_week1=(week1.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week1=total_flight_Week1/sum_cancellation_week1
print("Cancelation_ratio_Week1:",Cancelation_ratio_Week1)
####
week2=df.where(df['DAY_OF_WEEK'] == 2)#.head(3)
total_flight_Week2=week2.count()
sum_cancellation_week2=(week2.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week2=total_flight_Week2/sum_cancellation_week2
print("Cancelation_ratio_Week2:",Cancelation_ratio_Week2)
####
week3=df.where(df['DAY_OF_WEEK'] == 3)#.head(3)
total_flight_Week3=week3.count()
sum_cancellation_week3=(week3.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week3=total_flight_Week3/sum_cancellation_week3
print("Cancelation_ratio_Week3:",Cancelation_ratio_Week3)
####
week4=df.where(df['DAY_OF_WEEK'] == 4)#.head(3)
total_flight_Week4=week4.count()
sum_cancellation_week4=(week4.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week4=total_flight_Week4/sum_cancellation_week4
print("Cancelation_ratio_Week4:",Cancelation_ratio_Week4)
####
week5=df.where(df['DAY_OF_WEEK'] == 5)#.head(3)
total_flight_Week5=week5.count()
sum_cancellation_week5=(week4.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week5=total_flight_Week5/sum_cancellation_week5
print("Cancelation_ratio_Week5:",Cancelation_ratio_Week5)
####
week6=df.where(df['DAY_OF_WEEK'] == 6)#.head(3)
total_flight_Week6=week6.count()
sum_cancellation_week6=(week6.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week6=total_flight_Week6/sum_cancellation_week6
print("Cancelation_ratio_Week6:",Cancelation_ratio_Week6)
####
week7=df.where(df['DAY_OF_WEEK'] == 7)#.head(3)
total_flight_Week7=week7.count()
sum_cancellation_week7=(week7.agg({'CANCELLED': 'sum'}).collect()[0][0])
Cancelation_ratio_Week7=total_flight_Week7/sum_cancellation_week7
print("Cancelation_ratio_Week7:",Cancelation_ratio_Week7)



#df.withColumn("COUNT", lit(1)).groupBy("DAY_OF_WEEK", "CANCELLED").agg(func.sum("COUNT")).orderBy("sum(COUNT)", ascending=False).limit(5).show()


# Free the (nt)ontext object
sc.stop()

print("end")

