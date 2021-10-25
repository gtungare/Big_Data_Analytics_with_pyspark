# pyspark-template

This is a pyspark project template


## Describe here your project


I have used RDD appraoch

There are no special instruction to run except below

parameter 1 - gs://metcs777/TrainingData.txt
parameter 2 - gs://metcs777/TestingData.txt

Observation 

Task1 - Getting a good F1 score
Task2 - Getting F1 score of 0 , tried multiple options but still the same
Task3 -Logistic - F1 score significantly drop  dure to random choice tried couple of run but the same result 
Task3 -SVN - Getting F1 score of 0 , tried multiple options but still the same


# How to run  

Run the task 1 by submitting the task to spark-submit. 


```python

spark-submit main_task1.py gs://metcs777/TrainingData.txt gs://metcs777/TestingData.txt

```



```python

spark-submit main_task3.py gs://metcs777/TrainingData.txt gs://metcs777/TestingData.txt

```



```python

spark-submit main_task3_logistic.py gs://metcs777/TrainingData.txt gs://metcs777/TestingData.txt

```

```python

spark-submit main_task3_svm.py gs://metcs777/TrainingData.txt gs://metcs777/TestingData.txt

```

