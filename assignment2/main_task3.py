
# -*- coding: utf-8 -*-

"""
Name - Gaurav Tungare
Class: CS 777 - Fall 1
Date: Sep 2021
Homework  # Assignment 2 - Task 3 

thanks 

"""

import sys
from pyspark import SparkContext
from pyspark.sql import functions as func
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.sql import functions as F
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col


sc = SparkContext ( )

from pyspark.sql.types import *
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

#wikiCategoryFile="C:/Users/gaura/Downloads/CS777/assignment2/wiki-categorylinks-small.csv.bz2"
wikiCategoryFile="gs://metcs777/wiki-categorylinks.csv.bz2"

wikiCategoryLinks=sc.textFile(wikiCategoryFile)
wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"',''), x[1].replace('"','') ))
df=sqlContext.createDataFrame(wikiCats)

df.show()

"""
Task 3.1 (3 point)

The output is printed on console

"""

df_group_categ = df.groupBy (df[1]).count()
print("max_count_pages")
max_pages = df_group_categ.agg(func.max("count").alias('max_count_pages')).show()
print("avg_count_pages")
avg_pages = df_group_categ.agg(func.mean("count").alias('avg_count_pages')).show()
med = F.expr('percentile_approx(count, 0.5)')
median = df_group_categ.agg(med.alias('med_count)')).show()
print("stddev_pages")
std_pages = df_group_categ.select(_stddev(col('count')).alias('stddev_pages')).show()

"""
Task 3.2. (3 points)'
 top 10 mostly used wikipedia categories
 The output is printed on console
"""
print("top 10 mostly used wikipedia categories")
top_wikipedia_categories = df_group_categ.orderBy("count",ascending=[0]).show(10)
