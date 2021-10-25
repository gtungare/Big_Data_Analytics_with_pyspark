# pyspark-template

This is a pyspark project template


## Describe here your project


For Task 1, Task 2 and Task 3 are combined into a single script 

I have used RDD appraoch

There are no special instruction to run except below

parameter 1 - gs://metcs777/TrainingData.txt
parameter 2 - gs://metcs777/TestingData.txt

The dictonary from the training result is reused in test data set as explained by Prof in the class


Observation 

Despite changing Lamba for different learning rate, I see do not  see much change in the gradient value

Also the F1 score is coming out to be zero on the test data set. Even afte changing the lamba I do not see any improvement in the results 

 


# Submit your python scripts .py 

If your assignment has 3 tasks you need to commit the 3 scripts only and overwrite them. You can then delete the script number 4 ( main_task4.py 
)

If your assignment has 4 tasks then you can use all of them. 

# Other Documents. 

You can write your task description in this Markdown file or You can generate PDF file and added to the doc/ folder of your repository. 

Please note in your README.md file where your task description file is.  


# How to run  

Run the task 1 by submitting the task to spark-submit. 


```python

spark-submit main_task1_task2_task3.py 

parameter 1 - gs://metcs777/TrainingData.txt
parameter 2 - gs://metcs777/TestingData.txt

```







