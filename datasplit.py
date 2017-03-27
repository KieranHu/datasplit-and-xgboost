import numpy as np
import pandas as pd
import random


class Datasplit(object):
    def __init__(self, url):
        self.url = url
        self.important_variables = None
        self.split_method = [0.7, 0.1, 0.2]
        self.data = None
        self.train = None
        self.val = None
        self.test = None

    def set_data(self):
        self.data = pd.read_csv(self.url)

        return self.data

    def get_data(self):

        return self.train, self.val, self.test

    def set_important_variables(self, *important_variables):
        temp_lst = []
        for item in important_variables:
            temp_lst.append(item)
        self.important_variables = temp_lst

        return self.important_variables

    def get_important_variables(self):

        return self.important_variables

    def set_split_method(self, train = 0.7, val = 0.1, test = 0.2):
        self.split_method = [train, val, test]

        return self.split_method

    def get_split_method(self):

        return self.split_method

    def split_data(self):
        data = self.data
        variables = self.important_variables

        train = []
        validation = []
        test = []

        def scan(variables, list):
            if len(variables) == 0:
                size = len(list)
                train_size = round(size*self.split_method[0])
                validation_size = round(size*self.split_method[1])

                random.shuffle(list)
                train.extend(list[0:train_size])
                validation.extend(list[train_size: train_size+validation_size])
                test.extend(list[train_size+ validation_size: size])
                return

            select_var = variables.pop()
            values = data[select_var].unique()
            for val in values:
                df = data.ix[list]
                lst = df[df[select_var] == val].index.tolist()
                scan(variables, lst)
            variables.append(select_var)

        scan(variables, range(len(data)))

        self.train = self.data.ix[train]
        self.val = self.data.ix[validation]
        self.test = self.data.ix[test]

        return train, validation, test






