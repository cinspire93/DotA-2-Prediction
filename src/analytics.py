import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

class analyticsMachine(object):

    def __init__(self, models):
        '''
        This analytics class initializes with a list of models that the user wants to run on some dataframe
        '''
        self.models = models

    def full_pipeline(self, X, y, test_size=0.2, desired_cv_metric='accuracy', folds=5):
        '''
        Input: feature matrix X, labels y, desired test sample size, desired cross validation metric and number of folds
        Output: print out the cross validation score and test score
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model_and_score = {}
        for model in self.models:
            cross_validation_score = cross_val_score(model, X_train, y_train, scoring=desired_cv_metric, cv=folds, n_jobs=-1).mean()
            model_and_score[model] = cross_validation_score
            print "{:_^50}".format(model.__class__.__name__)
            print "The cross validation score for {} model is: {}".format(model.__class__.__name__, cross_validation_score)
            print "--------------------------------------------------\n"
        best_model = max(self.models, key=model_and_score.get)
        test_score = best_model.score(X_test, y_test)
        print "The test score for {} (best model) is: {}".format(best_model.__class__.__name__, test_score)
