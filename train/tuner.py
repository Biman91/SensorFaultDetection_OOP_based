
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


class Model_Finder():
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_random_forest(self, train_X, train_y):
        self.logger_object.log(self.file_object, "Entered get_best_params_for_random_forest method")
        try:
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5, verbose=3)
            self.grid.fit(train_X, train_y)

            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            self.clf.fit(train_X, train_y)
            self.logger_object.log(self.file_object, f"Random Forest best params: {self.grid.best_params_}")
            return self.clf
        except Exception:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_for_random_forest method')
            self.logger_object.log(self.file_object, 'Random Forest Parameter tuning  failed')
            raise Exception()

    def get_best_params_for_xgboost(self, train_X, train_y):
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_xgboost method')
        try:
            self.param_grid_xgboost = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]}
            self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3,
                                     cv=5)
            self.grid.fit(train_X, train_y)

            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.xgb = XGBClassifier(learning_rate=1, max_depth=5, n_estimators=50)
            self.xgb.fit(train_X, train_y)
            self.logger_object.log(self.file_object, f'XGBoost best params: {self.grid.best_params_}')
            return self.xgb
        except Exception:
            self.logger_object.log(self.file_object, "Exception occured in get_best_params_for_xgboost method")
            self.logger_object.log(self.file_object, "XGBoost Parameter tuning  failed")
            raise Exception()

    def get_best_model(self, train_X, train_y, test_X, test_y):
        self.logger_object.log(self.file_object, 'Entered the get_best_model method')
        try:
            self.xgboost = self.get_best_params_for_xgboost(train_X, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_X)
            if len(test_y.unique()) == 1:
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, f'Accuracy for XGBoost: {self.xgboost_score}')
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, f'AUC for XGBoost: {self.xgboost_score}')

            self.random_forest = self.get_best_params_for_random_forest(train_X, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_X)
            if len(test_y.unique()) == 1:
                self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
                self.logger_object.log(self.file_object, f'Accuracy for RF: {self.random_forest_score}')
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest)
                self.logger_object.log(self.file_object, f'AUC for RF: {self.random_forest_score}')

            if (self.random_forest_score < self.xgboost_score):
                return 'XGBoost', self.xgboost
            else:
                return 'RandomForest', self.random_forest
        except Exception:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_model method')
            self.logger_object.log(self.file_object, 'Model Selection Failed')
            return Exception()

