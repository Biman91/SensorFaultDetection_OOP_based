
from sklearn.model_selection import train_test_split
from train.logger import App_Logger
from train.train_data_loader import Data_Getter
from train.preprocessing import Preprocessor
from train.clustering import KMeansClustering
from train.tuner import Model_Finder
from train.fileOperation import File_Operation


class trainModel:
    def __init__(self):
        self.log_write = App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", "a+")

    def trainingModel(self):
        self.log_write.log(self.file_object, "Start of training")
        try:
            # load data
            data_getter = Data_Getter(self.file_object, self.log_write)
            data = data_getter.get_data()

            # Preprocessing
            preprocessor = Preprocessor(self.file_object, self.log_write)
            data = preprocessor.remove_columns(data, ['Wafer'])

            X, y = preprocessor.seperate_label_feature(data, label_column_name='Output')

            is_null_present = preprocessor.is_null_present(X)

            if is_null_present:
                X = preprocessor.impute_missing_values(X)

            column_to_drop = preprocessor.get_column_with_zero_std_dev(X)

            X = preprocessor.remove_columns(X, column_to_drop)

            # applying cluster Approach
            kmeans = KMeansClustering(self.file_object, self.log_write)
            number_of_clusters = kmeans.elbow_plot(X)

            X = kmeans.create_clusters(X, number_of_clusters)

            X['Labels'] = y

            list_of_clusters = X['Cluster'].unique()

            # cluster looking for best algorithm
            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]

                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                X_train, X_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                    random_state=355)

                model_finder = Model_Finder(self.file_object, self.log_write)
                best_model_name, best_model = model_finder.get_best_model(X_train, y_train, X_test, y_test)

                file_op = File_Operation(self.file_object, self.log_write)
                save_model = file_op.save_model(best_model, best_model_name + str(i))

            self.log_write.log(self.file_object, "Training Successfull")
            self.file_object.close()

        except Exception:
            self.log_write.log(self.file_object, "Training Unsuccessful")
            self.file_object.close()
            raise Exception()