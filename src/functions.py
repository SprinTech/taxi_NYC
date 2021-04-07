from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
import pandas as pd

def get_alpha_ridge(alpha_range, xtrain,ytrain):
    """ Use Ridge model to return the best alpha among the number of possibilities defined after 1000 iteration through the alpha range""""
    clf = RandomizedSearchCV(estimator=linear_model.Ridge(), 
                        param_distributions = {'alpha': range(alpha_range)},
                        n_iter = 1000,
                        cv=5,
                        refit=True,
                        error_score=0,
                        n_jobs=-1)

    clf.fit(xtrain, ytrain)
    return clf.best_estimator_.alpha


def get_alpha_lasso(alpha_range, xtrain,ytrain):
    """ Use Lasso model to return the best alpha among the number of possibilities defined after 1000 iteration through the alpha range""""
    clf = RandomizedSearchCV(estimator=linear_model.Lasso(), 
                        param_distributions = {'alpha': range(alpha_range)},
                        n_iter = 1000,
                        cv=5,
                        refit=True,
                        error_score=0,
                        n_jobs=-1)

    clf.fit(xtrain, ytrain)
    return clf.best_estimator_.alpha