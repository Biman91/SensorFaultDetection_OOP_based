
import pickle
import os
import shutil


class File_Operation:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory = 'models/'

    def save_model(self, model, filename):
        self.logger_object.log(self.file_object, "Entered the save model in file operation class")
        try:
            path = os.path.join(self.model_directory, filename)
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path + '/' + filename + '.sav', 'wb') as f:
                pickle.dump(model, f)
            self.logger_object.log(self.file_object, f"model file {filename} saved")
            return 'success'
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in save_model method")
            self.logger_object.log(self.file_object, f"model file {filename} not saved")
            raise Exception()

    def load_model(self, filename):
        self.logger_object.log(self.file_object, 'Entered the load_model method')
        try:
            with open(self.model_directory + filename + '/' + filename + '.sav', 'rb') as f:
                self.logger_object.log(self.file_object, f"model file {filename} loaded")
                return pickle.load(f)
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in load_model method of the Model_Finder class")
            self.logger_object.log(self.file_object, f"model file {filename} not saved")
            raise Exception()

    def find_correct_model(self, cluster_number):
        self.logger_object.log(self.file_object, 'Entered the find_correct_model_file method')
        try:
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_models_files = []
            self.list_of_files = os.listdir(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if (self.file.index(str(self.cluster_number)) != -1):
                        self.model_name = self.file
                except:
                    continue
            self.model_name = self.model_name.split('.')[0]
            self.logger_object.log(self.file_object, 'Exited the find_correct_model_file method')
            return self.model_name
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in find_correct_model_file method")
            self.logger_object.log(self.file_object, f"Exited the find_correct_model_file method")
            raise Exception()