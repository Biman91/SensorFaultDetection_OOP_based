import pandas as pd


class Data_Getter:
    def __init__(self, file_object, logger_object):
        self.training_file = 'TrainingFile/InputFile.csv'
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        self.logger_object.log(self.file_object, "Entered get_data method")
        try:
            self.data = pd.read_csv(self.training_file)
            self.logger_object.log(self.file_object, "Data load successfully")
            return self.data
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured get_data")
            self.logger_object.log(self.logger_object, "Data load unsuccessfully")