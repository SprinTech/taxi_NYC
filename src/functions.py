from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
import pandas as pd

def get_alpha_ridge(alpha_range, xtrain,ytrain):
    clf = RandomizedSearchCV(estimator=linear_model.Ridge(), 
                        param_distributions = {'alpha': range(alpha_range)},
                        n_iter = 1000,
                        cv=5,
                        refit=True,
                        error_score=0,
                        n_jobs=-1)

    clf.fit(xtrain, ytrain)
    return clf.best_estimator_.alpha
    # df1 = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    # get_alpha_ridge.max_alpha = df1['param_alpha'].index[0]
    # print("L'alpha optimal pour la range indiquée est {}".format(get_alpha_ridge.max_alpha))



def get_alpha_lasso(alpha_range, xtrain,ytrain):
    clf = RandomizedSearchCV(estimator=linear_model.Lasso(), 
                        param_distributions = {'alpha': range(alpha_range)},
                        n_iter = 1000,
                        cv=5,
                        refit=True,
                        error_score=0,
                        n_jobs=-1)

    clf.fit(xtrain, ytrain)
    return clf.best_estimator_.alpha
    # df1 = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    # get_alpha_lasso.max_alpha = df1['param_alpha'].index[0]
    # print("L'alpha optimal pour la range indiquée est {}".format(get_alpha_lasso.max_alpha))