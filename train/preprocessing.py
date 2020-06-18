
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class Preprocessor:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        self.logger_object.log(self.file_object, "Entered remove_columns method")
        self.data = data
        self.columns = columns
        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)
            self.logger_object.log(self.file_object, "Column removal successful")
            return self.useful_data
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured")
            self.logger_object.log(self.file_object, "Unsuccessful to remove column")
            raise Exception()

    def seperate_label_feature(self, data, label_column_name):
        self.logger_object.log(self.file_object, "Enter Separate labeled column")
        try:
            self.X = data.drop(labels=label_column_name, axis=1)
            self.y = data[label_column_name]
            self.logger_object.log(self.file_object, "Label separation successful")
            return self.X, self.y
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured label_column_name")
            self.logger_object.log(self.file_object, "Label Separation unsuccessful")
            raise Exception()

    def is_null_present(self, data):
        self.logger_object.log(self.file_object, "Entered is_null_present method")
        self.null_present = False
        try:
            self.null_counts = data.isna().sum()
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if self.null_present:
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing_value_count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv("trainOutput/null_values.csv")
            self.logger_object.log(self.file_object, "Finding missing value is success")
            return self.null_present
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in is_null_present method")
            self.logger_object.log(self.file_object, "Finding missing value Failed")
            raise Exception()

    def impute_missing_values(self, data):
        self.logger_object.log(self.file_object, "Entered impute_missing_value method")
        self.data = data
        try:
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger_object.log(self.file_object, "Imputed missing values are success")
            return self.new_data
        except Exception:
            self.logger_object.log(self.file_object, "Eror occured on impute_missing_values")
            self.logger_object.log(self.file_object, "Imputation unsuccessful")
            raise Exception()

    def get_column_with_zero_std_dev(self, data):
        self.logger_object.log(self.file_object, "Entered get_column_with_zero_std_dev method")
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            for i in self.columns:
                if self.data_n[i]['std'] == 0:
                    self.col_to_drop.append(i)
            self.logger_object.log(self.file_object, "column search for std is successful")
            return self.col_to_drop
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in get_column_with_zero_std_dev method")
            self.logger_object.log(self.file_object, "column search for std is unsuccessful")
            raise Exception()