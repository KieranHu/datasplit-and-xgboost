# Describe

The code is split into three different python file: *demo.py*, *datasplit.py* and  *algorithm.py*. The file *algorithm.py* contains the machine learning algorithm to do classification.
The file *datasplit.py* contains the required method to split the dataset.

# Algorithm

The algorithm I used is called gradient boosting tree. It is a kind of decision tree machine learning method that each step based on previous step's error rate. In detail, firstly, we build a tree and compute the predict error, then we build another tree based on the predict error until reach the error rate we want.

Here is same important parameters for this algorithm:

*eta*: learning rate  
*lambda*: L2 regularization term  
*alpha*: L1 regularization term  
*max_depth*: max depth of the tree  
*subsample*: subsample ratio of the training instance  
*colsample_bytree*: subsample ratio of columns when constructing each tree

Reference: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

# Datasplit algorithm

Basically, I use recursive method to find the indexes of each subclass. Then do split on each subclass.  In detail, firstly, I scan the whole important features. Then I pick up the first important variables and find all its possible value. I , then, sub group the data according to the possible values. For example, say var1 is important variable. It has four possible value, A ,B, C, D. The group the dataset into four sub dataset. For sub1 dataset, var1 only have value A, and sub2 dataset only have value B and so on. Then for each sub dataset considering important variables var2 and subgroup each sub dataset according to values in var2. Following this step for all important variables. When we finish, we have several sub groups. Then randomly split each group according to split rate and combine them together to form train, validation and test dataset. If use for loop to do this algorithm, it will be very computational costly, since the computational cost will be O(n^p). n is the number of rows in the dataset and p is the number of important variables. 

# How to

### Run the code in *demo.py*. To run this code, python3 is required as well as python packages xgboost, sklearn, pandas, numpy and random.

### Input the file location into the Datasplit class.

*set_data()* return input data in pandas dataFrame.  

*spit_data()* return  index for train, validation and test dataset.

*set_important_variables(\* important_variables)* input column names for important.   variables, return list for important variables.  

*get_important_variables()* return list for important variables.

*get_data()* return split data in pandas dataFrame.  

*set_split_method(train, val, test)* input split ratio for train, validation and test dataset. Default is [0.7, 0.1, 0.2]. Return split ratio list.  

*get_split_method()* return split ratio list.  

### Input the train, validation, test dataset into Classification class.

*set_param(param)* input a dictionary for parameters, return the dictionary for parameters.  

*get_param()* return the dictionary for parameters.  

*set_label_name(name)* input the column name for response variables, return the column name.  

*get_label_name()* return the column name for response variables.  

*train_model(num_round)* input number of round, default is 5, return the xgboost model.  

# Big O
The computational cost for data split is O(n).  
