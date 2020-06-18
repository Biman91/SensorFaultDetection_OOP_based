import pandas as pd


class Data_getter_pred:
    def __init__(self, file_object, logger_object):
        self.prediction_file = 'predictionFile/InputFile.csv'
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        self.logger_object.log(self.file_object, "Entered the get_data method")
        try:
            self.data = pd.read_csv(self.prediction_file)
            self.logger_object.log(self.file_object, "Data Load Successful")
            return self.data
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in get_data method")
            self.logger_object.log(self.file_object, "Data Load Unsuccessful")
            raise Exception()

    def delete_existing_prediction_input(self):
        self.logger_object.log(self.file_object, "Entered the delete_existing_prediction_input method")
        if os.path.exists('predictionFile/InputFile.csv'):
            os.remove('predictionFile/InputFile.csv')


