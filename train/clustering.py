# Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from train.fileOperation import File_Operation



class KMeansClustering:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self, data):
        self.logger_object.log(self.file_object, "Enterened elbow_plot method")
        wcss = []
        try:
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, 11), wcss)
            plt.title("The ELBOW method")
            plt.xlabel("No of cluster")
            plt.ylabel("WCSS")
            plt.savefig("trainOutput/K-means Elbow.png")
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(self.file_object, f"Optimum number of cluster is {self.kn.knee}")
            return self.kn.knee
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in elbow_plot method")
            self.logger_object.log(self.file_object, "Finding the number of clusters failed")
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        self.logger_object.log(self.file_object, "Entered create_clusters method")
        self.data = data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            self.y_means = self.kmeans.fit_predict(data)
            self.file_op = File_Operation(self.file_object, self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans')
            self.data['Cluster'] = self.y_means
            self.logger_object.log(self.file_object, "Successfully created cluster")
            return self.data
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in create_clusters method")
            self.logger_object.log(self.file_object, "Fitting the data to clusters failed")
            raise Exception()