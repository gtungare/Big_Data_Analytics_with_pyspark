# pyspark-template

This is a pyspark  template used to describe my assignments 


## Describe here your project


Following is the logic used in main_task1.py  

Step 1 - Check for number of input paramenters . As mentioned in the class there are two paramenters expected 1. input file path 2. output folder
Step 2 - I am using spark dataframe to efficiently read the csv file , using read.csv option to read the input csv file 
step 3 - I am reading the fist two columns which are the medallion  and hack_license to be used for the calculation 
step 4 - The logic uses group by medallion, followed by count of the hack_license
step 5 - Using the where statement followed by tail to get the top 10 active taxis

Following is the logic used in main_task2.py

Step 1 - Check for number of input paramenters . As mentioned in the class there are two paramenters expected 1. input file path 2. output folder
Step 2 - I am using spark dataframe to efficiently read the csv file , using read.csv option to read the input csv file 
step 3 - I am reading the fist two columns which are the hack_license,trip_time_in_secs  and total_amount to be used for the calculation 
step 4 - The logic calcualtes new column money_per_minute based on total_amount/trip_time_in_secs/60 , 60 is used in the calculation since the time is based in secs.
step 5 - Using the where statement followed by head to get the top 10 best drivers. Please note I have used rounding upto 3 digits 

# Other Documents. 

I have attached gtungare_assignment_1 which has the details of the screenshot. Please see attached referece doc


# How to run  

Run the task 1 by submitting the task to spark-submit. 


```python

spark-submit main_task1.py 

THe following input was used for running task 1 

Input 

gs://metcs777/taxi-data-sorted-large.csv.bz2

output 

gs://gtungare_bucket_1/outputq1_large1


```



```python

spark-submit main_task2.py 

THe following input was used for running task 1 

Input 

gs://metcs777/taxi-data-sorted-large.csv.bz2

output 

gs://gtungare_bucket_1/outputq1_large1

```




