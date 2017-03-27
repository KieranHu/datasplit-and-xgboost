import pandas as pd
from datasplit import Datasplit
from algorithm import Classification

# Set url
file_location = 'testdata.csv'
Data = Datasplit(file_location)
data = Data.set_data()

# Set important variables
Data.set_important_variables('var1', 'var2')

# Set split method
Data.set_split_method(train = 0.7, val = 0.1, test = 0.2)

# Get data
Data.split_data()
train, validation, test = Data.get_data()


# Input train, validation and test dataset into classification algorithm
classification = Classification(train, validation, test)

# Set parameters
# For detail, looking at https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
param = {'eta': 0.5, 'seed': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic', 'max_depth':8, 'min_child_weight':4}
classification.set_param(param)

# Set response variable
classification.set_label_name('outcome')

# Train the model
classification.train_model(num_round = 30)

