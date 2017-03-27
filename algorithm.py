import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

class Classification(object):
    def __init__(self, train, validation, test):
        self.train_data= train
        self.val_data = validation
        self.test_data = test
        self.param = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}
        self.label_name = None

    def set_param(self, param):
        self.param = param

        return self.param

    def get_param(self):
        return self.param

    def set_label_name(self, name):
        self.label_name = name

        return self.label_name

    def get_label_name(self):

        return self.label_name

    def train_model(self, num_round = 5):

        """
        :param num_round:
        :return: model
        """

        """
        Here is the part for data clean
        """
        train_data = self.train_data
        val_data = self.val_data
        test_data = self.test_data

        def data_clean(data):
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = pd.Categorical(data[col]).codes
            return data

        train_data = data_clean(train_data)
        val_data = data_clean(val_data)
        test_data = data_clean(test_data)

        train_target = train_data.pop(self.label_name)
        val_target = val_data.pop(self.label_name)
        test_target = test_data.pop(self.label_name)

        dtrain = xgb.DMatrix(train_data, label = train_target)
        dval = xgb.DMatrix(val_data, label = val_target)
        dtest = xgb.DMatrix(test_data)

        """
        Train the model
        """
        param = self.param
        evallist = [(dval,'eval'), (dtrain,'train')]

        model = xgb.train(params = param, dtrain = dtrain, num_boost_round=num_round, evals = evallist)


        """
        See the accuracy on test data set
        """
        predict = model.predict(dtest)
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        accuracy = accuracy_score(predict, test_target)
        error_rate = 1 - accuracy
        print("Accuracy is: %.2f"%(accuracy))
        print("Error rate is: %.2f"%(error_rate))

        return model