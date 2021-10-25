"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Module 3 Task 1

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
    if len(sys.argv) != 2:
        print(" Please check the arguments <file> <output> ", file=sys.stderr)
        exit(-1)

# Print the script name
print("PySpark Script: ", sys.argv[0])
sc = SparkContext(appName="Module3_Task1")

# Create a spark context and print some information about the context object

conf = SparkConf().setAppName("Process_mod3_q1")
sqlContext = sql.SQLContext(sc)
df1 = sqlContext.read.csv(sys.argv[1])

# Filtering the record per of taking total amount between 1 and 600

dfn=df1.select(['_c5','_c11']).where((df1['_c16'] < 600.00 ) & (df1['_c16'] > 1.00 ) ).toDF("trip_distance","total_amount")
dfn=dfn.withColumn("xi_mul_yi",dfn["trip_distance"]*dfn["total_amount"])
dfn=dfn.withColumn("xi_square",dfn["trip_distance"]*dfn["trip_distance"])
n=dfn.count()
print(n)

sum_xi_yi=(dfn.agg({'xi_mul_yi': 'sum'}).collect()[0][0])
print("n_sum_xi_yi=",sum_xi_yi)
sum_xi=dfn.agg({'trip_distance': 'sum'}).collect()[0][0]
print("sum_xi=",sum_xi)
sum_yi=dfn.agg({'total_amount': 'sum'}).collect()[0][0]
print("sum_yi=",sum_yi)
sum_xi_square=(dfn.agg({'xi_square': 'sum'}).collect()[0][0])
print("um_xi_square=",sum_xi_square)
square_sum_xi=(sum_xi)*(sum_xi)
print("square_sum_xi=",square_sum_xi)
### calculation for slope
m_slope=((n*sum_xi_yi)-(sum_xi*sum_yi))/((n*sum_xi_square)-(square_sum_xi))
print("m_slope=",m_slope)
b_hat=(((sum_xi_square)*(sum_yi))-((sum_xi)*(sum_xi_yi)))/((n*sum_xi_square)-(square_sum_xi))
print("b_hat=",b_hat)

"""
# now we want to store the result in a single file on the cluster
#dataToASingleFile=sc.parallelize(top_10_active_taxis).coalesce(1)

#dataToASingleFile.saveAsTextFile(sys.argv[2])

#print(top_10_active_taxis)
"""

# Free the (nt)ontext object
sc.stop()

print("end")
