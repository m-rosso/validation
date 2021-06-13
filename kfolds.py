####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np
import json
import os

from datetime import datetime
import time
import progressbar

from scipy.stats import uniform, norm, randint

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss, mean_squared_error

# pip install lightgbm
import lightgbm as lgb
# pip install xgboost
import xgboost as xgb

from utils import running_time, cross_entropy_loss

from concurrent.futures import ThreadPoolExecutor

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

"""
This module contains two classes:
	1) "KfoldsCV": executes K-folds cross-validation combined with grid or random searches for defining hyper-parameters of several
	distinct machine learning algorithms.

	2) "KfoldsCV_fit": extends the first to include the estimation of a model using the entire training data and the best choices of
	hyper-parameters, in addition to providing estimates of performance metrics evaluated on test data. This class inherits from
	"KfoldsCV", mainly because both initializations are the same. The main differences between these classes are that the first
	does not allow fitting a final model, and that no test data is provided to it.

Both classes are specially relevant for empirical contexts where available data has no sufficient length to provide appropriate
batches of training, validation, and test data. Consequently, the fine tuning of hyper-parameters should be implemented based solely
on the training data.

K-folds cross-validation allows appropriate predictions for training data instances, yet its main application consists of exploring
different combinations of values for relevant hyper-parameters. There are two main alternatvies for performing fine tuning under
this validation procedure: grid search and random search. Both are available here by making use of "grid_param" and "random_search"
initialization parameters.

Setting "random_search" to False and declaring "grid_param" as a dictionary whose keys are the most relevant hyper-parameters of a
ML method and whose respective values are lists with values, one can execute grid search, which is more likely to provide fine
results when previous studies have been done so only most promising alternatives remain left.

Setting "random_search" to True and declaring "grid_param" as a dictionary whose keys are the most relevant hyper-parameters of a
ML method and whose respective values are lists with integer values or statistical distributions from scipy.stats (such as "uniform",
"norm", "randint"), then random search can be implemented. This alternative is specially suited for ML methods that have multiple
hyper-parameters.

Supervised learning tasks available using these classes are binary classification and regression. Some slight modifications are
required for implementing multiclass classification (which should be done soon). For these two problems, the following ML methods
are supported:
	1) Logistic regression (from sklearn).
		* Hyper-parameters for tuning: regularization parameter ('C').
	2) Linear regression (Lasso) (from sklearn).
		* Hyper-parameters for tuning: regularization parameter ('C').
	3) GBM (sklearn).
		* Hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('learning_rate'),
        number of estimators ('n_estimators').
    4) GBM (LightGBM).
        * Hyper-parameters for tuning: subsample ('bagging_fraction'), maximum depth ('max_depth'), learning rate ('learning_rate'),
        number of estimators ('num_iterations').
    5) GBM (XGBoost).
        * Hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('eta'), number of
        estimatores ('num_boost_round').
	6) Random forest (from sklearn).
		* Hyper-parameters for tuning: number of estimators ('n_estimators'), maximum number of features ('max_features') and minimum
        number of samples for split ('min_samples_split').
	7) SVM (from sklearn).
		* Hyper-parameters for tuning: regularization parameter ('C') kernel ('kernel'), polynomial degree ('degree'), gamma
        ('gamma').

Performance metrics allowed for binary classification are ROC-AUC, average precision score (as proxy for precision-recall AUC), and
Brier score. For regression, RMSE is the metric available by now.

Declaring "parallelize" equal to True when initializing the K-folds CV object is expected to improve running time, since
training-validation estimation for all K folds of data is then implemented in parallel.

Recognizing that sklearn already offers its own classes and functions for cross-validation, grid search, and random search, these
are the main features that may supplement the use of sklearn:
	1) Simplified usage: its interface is maybe more intuitive, since it requires only the creation of an object from "KfoldsCV" or
	from "KfoldsCV_fit". All relevant parameters are then initialized, and only a method for running/fitting should be executed
	having as arguments only the data. Finally, all relevant outcomes from executing these validation methods are kept together with
	the object as attributes.

	2) Additional features: more information is provided by the end of estimations, such as cross-validation predictions for
	training data, performance metrics, progress bar and elapsed time.

	3) Pre-selection of features: the use of statistical learning methods (here, logistic regression or Lasso) for features selection
	is available. Even that all ML methods have some kind of regularization, this should help filtering out non-relevant features
	for some specific empirical context. If this kind of features selection is expected to happen in final model estimation, it
	should also be internatilized during definition of hyper-parameters.

	4) More flexibility: by changing components of method "__create_model", models from any library can be applied, not only those
	provided by sklearn, all in the same framework. Currently, Light GBM is available, but also neural networks from Keras should
    probably be inserted soon.
"""
####################################################################################################################################
# K-folds cross-validation for grid/random search:

class KfoldsCV(object):
    """
    Arguments for initialization:
        :param task: distinguishes between classification and regression supervised learning tasks. It should be
        coherent with "method" argument.
        :type task: string.
        
        :param method: indicates which machine learning method should be applied for K-folds CV estimation. Check
        documentation of the module for available methods.
        :type method: string.
        
        :param metric: provides the performance metric for guiding grid search or random search. Check
        documentation of the module for available metrics.
        :type method: string.
        
        :param num_folds: number of folds for cross-validation.
        :type num_folds: integer (greater than zero).
        
        :param shuffle: defines whether training data should be randomized before split.
        :type shuffle: boolean.
        
        :param pre_selecting: defines whether pre-selection of features should take place at each K-folds CV
        iteration.
        :type pre_selecting: boolean.
        
        :param pre_selecting_param: regularization parameter of methods for pre-selecting features.
        :type pre_selecting_param: float (greater than zero).
        
        :param random_search: defines whether random search should be executed instead of grid search.
        :type random_search: boolean.
        
        :param n_samples: number of samples for random search.
        :type n_samples: integer (greater than zero).
        
        :param grid_param: values that should be tested for each hyper-parameter.
        :type grid_param: dictionary (strings as keys and lists as values).
        
        :param default_param: values of hyper-parameter for default. When no grid search or random search should
        occur, or when error emerges during all iterations of K-folds CV estimation, then this collection of
        hyper-parameters is finally displayed as chosen.
        :type default_param: dictionary (strings as keys and floats/integers/strings as values).
    
        :param parallelize: indicates whether K-folds estimation should be parallelized, running K estimations at once.
        :type parallelize: boolean.
    
    Methods:
        "run": executes the K-folds CV estimation together with grid search or random search.
        
        "create_grid": this static method creates a list with dictionaries whose keys are hyper-parameters and whose
        values are predefined during initialization. Elements of this list are combinations of values for all
        hyper-parameters declared in "grid_param".
        
        "pre_selection": this static method executes a (linear) ML algorithm which finally selects a subset of
        features to be kept for model estimation.
    
    Output objects:
        "CV_scores": dataframe with K-folds CV predicted values for training data associated with best parameters.
        "CV_metric": dataframe with performance metric by combination of hyper-parameters.
        "CV_selected_feat": list with pre-selected features by iteration.
        "best_param": dictionary with the best value for each hyper-parameter.
        "running_time": overall running time.
    """
    
    # Dictionary with functions for computing performance metrics:
    __metrics_functions = {
        'roc_auc': roc_auc_score,
        'avg_precision_score': average_precision_score,
        'brier_loss': brier_score_loss,
        'mse': mean_squared_error,
        'cross_entropy': cross_entropy_loss
    }

    def __init__(self, task='classification', method='logistic_regression',
                 metric='roc_auc', num_folds=3, shuffle=False,
                 pre_selecting=False, pre_selecting_param=None,
                 random_search=False, n_samples=None,
                 grid_param=None, default_param=None,
                 parallelize=False):
        self.task = task
        self.method = str(method)
        self.metric = metric
        self.num_folds = int(num_folds)
        self.shuffle = shuffle
        self.default_param = default_param
        self.pre_selecting = pre_selecting
        self.pre_selecting_param = pre_selecting_param
        self.random_search = random_search
        self.n_samples = n_samples
        self.parallelize = parallelize
        
        # Creating a list with combinations of values for hyper-parameters:
        self.grid_param = self.create_grid(params=grid_param, random_search=self.random_search,
                                           n_samples=self.n_samples)
    
    # Method for running Kfolds CV using declared inputs and output:
    def run(self, inputs, output, progress_bar=True, print_outcomes=True, print_time=True):
        """
        Method for running Kfolds CV using declared inputs and output.
        
        :param inputs: inputs of training data.
        :type inputs: dataframe.

        :param output: values for response variable of training data.
        :type output: dataframe.

        :param progress_bar: defines whether progress bar should be printed during execution.
        :type progress_bar: boolean.

        :param print_outcomes: defines whether final outcomes should be printed after execution.
        :type print_outcomes: boolean.

        :param print_time: defines whether total elapsed time should be printed after execution.
        :type print_time: boolean.
        """
        # Registering start time:
        start_time = datetime.now()
        
        # Splitting training data into 'k' different folds of data:
        k = list(range(self.num_folds))
        k_folds_X, k_folds_y = self.__splitting_data(inputs=inputs, output=output, num_folds=self.num_folds,
                                                     shuffle=self.shuffle)
        
        # Object for storing selected features (if self.pre_select = True):
        if self.pre_selecting:
            self.CV_selected_feat = {}
        
        # Object for storing performance metrics for each combination of hyper-parameters:
        self.CV_metric = pd.DataFrame()
        
        # Object for storing CV estimated scores:
        CV_scores = dict(zip([str(g) for g in self.grid_param],
                             [pd.DataFrame(data=[],
                                           columns=['cv_score']) for i in range(len(self.grid_param))]))

        # Progress bar measuring grid estimation progress:
        if progress_bar:
            bar_grid = progressbar.ProgressBar(maxval=len(self.grid_param), widgets=['\033[1mGrid estimation progress:\033[0m ',
                                                                                    progressbar.Bar('-', '[', ']'), ' ',
                                                                                    progressbar.Percentage()])
            bar_grid.start()
        
        # Loop over combinations of hyper-parameters:
        for j in range(len(self.grid_param)):
            # Object for storing performance metrics for each combination of hyper-parameters:
            CV_metric_list = []
            
            # Implementing train-validation estimation and assessing performance metrics and predicted values:
            try:
                if self.parallelize:
                    # Creating list of codes to run in parallel:
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(self.__run_estimation, k_folds_X, k_folds_y, i,
                                                   j, CV_scores) for i in k]

                        # Executing codes:
                        for f in futures:
                            estimation = f.result()
                            CV_metric_list.append(estimation['metrics'])
                            CV_scores[str(self.grid_param[j])] = estimation['preds']
                
                else:
                    # Loop over folds of data:
                    for i in k:
                        estimation = self.__run_estimation(folds_X=k_folds_X, folds_y=k_folds_y, fold_idx=i,
                                                           param_idx=j, CV_scores_dict=CV_scores)
                        CV_metric_list.append(estimation['metrics'])
                        CV_scores[str(self.grid_param[j])] = estimation['preds']
                
                # Dataframes with CV performance metrics:
                self.CV_metric = pd.concat([self.CV_metric,
                                            pd.DataFrame(data={'tun_param': str(self.grid_param[j]),
                                                               'cv_' + self.metric: np.nanmean(CV_metric_list)},
                                                         index=[j])], axis=0, sort=False)

            except Exception as e:
                print('\033[1mNot able to perform CV estimation with parameters ' +
                      str(self.grid_param[j]) + '!\033[0m')

                # Dataframes with CV performance metrics:
                self.CV_metric = pd.concat([self.CV_metric,
                                            pd.DataFrame(data={'tun_param': str(self.grid_param[j]),
                                                               'cv_' + self.metric: np.NaN},
                                                         index=[j])], axis=0, sort=False)

                print(f'Error: {e}')

            if progress_bar:
                bar_grid.update(j+1)
                time.sleep(0.1)
        
        # Best tuning hyper-parameters:
        try:
            if (self.metric == 'brier_loss') | (self.metric == 'mse'):
                self.best_param = self.CV_metric['cv_' + self.metric].idxmin()
            else:
                self.best_param = self.CV_metric['cv_' + self.metric].idxmax()
                
            self.best_param = self.grid_param[self.best_param]
            self.__using_default = False
            
        except:
            self.best_param = self.default_param
            self.__using_default = True
        
        # CV scores for best tuning parameter:
        try:
            self.CV_scores = CV_scores[str(self.best_param)]

        except:
            self.CV_scores = pd.DataFrame(data={
                'cv_scores': [np.NaN for i in range(len([y for l in k_folds_y for y in l]))],
                'y_true': [y for l in k_folds_y for y in l]
            })
            
        # Registering end time of K-folds CV:
        end_time = datetime.now()

         # Printing outcomes:
        if print_outcomes:
            self.__print_outcomes()
            print('\n')
        
        self.running_time = running_time(start_time=start_time, end_time=end_time, print_time=print_time)
        
    # Function that creates a list with different combinations of hyper-parameters:
    @staticmethod
    def create_grid(params, random_search, n_samples=10):
        """
        Creates a list with dictionaries whose keys are hyper-parameters and whose values are predefined during
        initialization.
        
        :param params: dictionary whose keys are hyper-parameters and values are lists with values for testing.
        :type params: list.
        
        :param random_search: defines whether random search should be executed instead of grid search.
        :type random_search: boolean.
        
        :param n_samples: number of samples for random search.
        :type n_samples: integer (greater than zero).
        
        :return: list with combinations of values for hyper-parameters.
        :rtype: list.
        """
        # Grid search:
        if not random_search:
            list_param = [params[k] for k in params.keys()]
            list_param = [list(x) for x in np.array(np.meshgrid(*list_param)).T.reshape(-1,len(list_param))]
            grid_param = []

            for i in list_param:
                grid_param.append(dict(zip(params.keys(), i)))

        # Random search:
        else:
            grid_param = []

            for i in range(1, n_samples+1):
                list_param = []

                for k in params.keys():
                    try:
                        list_param.append(params[k].rvs(1)[0])
                    except:
                        list_param.append(np.random.choice(params[k]))
                grid_param.append(dict(zip(params.keys(), list_param)))

        return grid_param
    
    # Function that splits data into K folds of data:
    def __splitting_data(self, inputs, output, num_folds, shuffle):
        if shuffle:
            indices = np.random.choice(range(len(inputs)), size=len(inputs), replace=False)

            X = np.array_split(inputs.reset_index(drop=True).reindex(index = indices), num_folds)
            y = np.array_split(output.reset_index(drop=True).reindex(index = indices), num_folds)

        else:
            X = np.array_split(inputs, num_folds)
            y = np.array_split(output, num_folds)

        return (X, y)

    # Function that runs train-validation estimation:
    def __run_estimation(self, folds_X, folds_y, fold_idx, param_idx, CV_scores_dict):
        # Train and validation split:
        X_train = pd.concat([x for l,x in enumerate(folds_X) if (l!=fold_idx)], axis=0, sort=False)
        y_train = pd.concat([x for l,x in enumerate(folds_y) if (l!=fold_idx)], axis=0, sort=False)
                
        X_val = folds_X[fold_idx]
        y_val = folds_y[fold_idx]

        # Prior selection of features:
        if self.pre_selecting:
            selected_features = self.pre_selection(input_train=X_train, output_train=y_train,
                                                   regul_param=self.pre_selecting_param,
                                                   task = self.task)
            self.CV_selected_feat[str(fold_idx+1)] = selected_features

            X_train = X_train[selected_features]
            X_val = X_val[selected_features]

        if self.method == 'light_gbm':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'bagging_fraction': float(self.grid_param[param_idx]['bagging_fraction']),
                     'learning_rate': float(self.grid_param[param_idx]['learning_rate']),
                     'max_depth': int(self.grid_param[param_idx]['max_depth']),
                     'num_iterations': int(self.grid_param[param_idx]['num_iterations']),
                     'bagging_freq': 1,
                     'metric': self.grid_param[param_idx].get('metric') if self.grid_param[param_idx].get('metric') else '',
                     'verbose': -1}

            # Defining dataset for light GBM estimation:
            train_data = lgb.Dataset(X_train.values, label = y_train.values)

            # Creating and training the model:
            model = lgb.train(param, train_data, 10, verbose_eval = False)

            # Predicting scores:
            score_pred = model.predict(X_val.values)
            
        elif self.method == 'xgboost':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'subsample': float(self.grid_param[param_idx]['subsample']),
                     'eta': float(self.grid_param[param_idx]['eta']),
                     'max_depth': int(self.grid_param[param_idx]['max_depth'])}
            
            # Creating the training and validation data objects:
            dtrain = xgb.DMatrix(data=X_train, label=y_train)
            dval = xgb.DMatrix(data=X_val)
            
            # Training the model:
            model = xgb.train(params=param, dtrain=dtrain,
                              num_boost_round=int(self.grid_param[param_idx]['num_boost_round']))
            
            # Predicting scores:
            score_pred = model.predict(dval)

        else:
            # Create the estimation object:
            model = self.__create_model(task=self.task, method=self.method, params=self.grid_param[param_idx])

            # Training the model:
            model.fit(X_train, y_train)

            # Predicting scores:
            if self.task == 'classification':
                score_pred = [p[1] for p in model.predict_proba(X_val)]

            else:
                score_pred = [p for p in model.predict(X_val)]

        # Calculating performance metric:
        perf_metrics = self.__metrics_functions[self.metric](y_val, score_pred)


        # Dataframes with CV scores:
        new_CV_scores = pd.DataFrame(data={'y_true': list(y_val), 'cv_score': score_pred},
                                     index=list(y_val.index))
        updated_CV_scores = pd.concat([CV_scores_dict[str(self.grid_param[param_idx])],
                                       new_CV_scores], axis=0, sort=False)

        return {
            'metrics': perf_metrics,
            'preds': updated_CV_scores
        }
    
    # Function that applies L1 regularized linear model (linear or logistic regression) to select features for
    # each estimation of K-folds CV:
    @staticmethod
    def pre_selection(input_train, output_train, regul_param, task='classification'):
        """
        Executes a (linear) ML algorithm which finally selects a subset of features to be kept for model
        estimation.
        
        :param input_train: inputs of training data.
        :type input_train: dataframe.

        :param output_train: values for response variable of training data.
        :type output_train: dataframe.
        
        :param pre_selecting_param: regularization parameter of methods for pre-selecting features.
        :type pre_selecting_param: float (greater than zero).
        
        :param task: distinguishes between classification and regression supervised learning tasks. It should be
        coherent with "method" argument.
        :type task: string.
        
        :return: list of pre-selected features.
        :rtype: list.
        """
        if (task == 'classification') | (task == 'binary'):
            model = LogisticRegression(solver='liblinear', penalty = 'l1',
                                       C = regul_param, max_iter = 100, tol = 1e-4)
            model.fit(input_train, output_train)
            betas = list(model.coef_[0])

        else:
            model = Lasso(alpha = regul_param, max_iter = 5000, tol = 1e-4)
            model.fit(input_train, output_train)
            betas = list(model.coef_)

        model_outcomes = pd.DataFrame(data={'feature': list(input_train.columns), 'beta': betas})

        return [f for f in list(model_outcomes[model_outcomes['beta']!=0].feature)]

    # Function that creates estimation objects:
    def __create_model(self, task, method, params):
        if task == 'classification':    
            if method == 'logistic_regression':
                model = LogisticRegression(solver='liblinear',
                                           penalty = 'l1',
                                           C = params['C'],
                                           warm_start=True)

            elif method == 'GBM':
                model = GradientBoostingClassifier(subsample = float(params['subsample']),
                                                   max_depth = int(params['max_depth']),
                                                   learning_rate = float(params['learning_rate']),
                                                   n_estimators = int(params['n_estimators']),
                                                   warm_start = True)

            elif method == 'random_forest':
                model = RandomForestClassifier(bootstrap = True, criterion = 'gini',
                                               n_estimators = int(params['n_estimators']),
                                               max_features = int(params['max_features']),
                                               min_samples_split = int(params['min_samples_split']),
                                               warm_start = True)

            elif method == 'SVM':
                model = SVC(C = float(params['C']),
                            kernel = params['kernel'],
                            degree = int(params['degree']),
                            gamma = params['gamma'],
                            probability = True, coef0 = 0.0, shrinking = True, tol = 0.001, max_iter = -1,
                            cache_size = 200, class_weight = None, decision_function_shape = 'ovr',
                            verbose = False, random_state = None)

        elif task == 'regression':    
            if method == 'lasso':
                model = Lasso(alpha=params['alpha'])

            elif method == 'GBM':
                model = GradientBoostingRegressor(subsample = float(params['subsample']),
                                                  max_depth = int(params['max_depth']),
                                                  learning_rate = float(params['learning_rate']),
                                                  n_estimators = int(params['n_estimators']),
                                                  warm_start = True)

            elif method == 'random_forest':
                model = RandomForestRegressor(bootstrap = True, criterion = 'mse',
                                              n_estimators = int(params['n_estimators']),
                                              max_features = int(params['max_features']),
                                              min_samples_split = int(params['min_samples_split']),
                                              warm_start = True)

            elif method == 'SVM':
                model = SVR(C = float(params['C']),
                            kernel = params['kernel'],
                            degree = int(params['degree']),
                            gamma = params['gamma'],
                            epsilon = 0.1, coef0 = 0.0, shrinking = True, cache_size = 200,
                            tol = 0.001, max_iter = -1, verbose = False)

        return model

    # Function that prints outcomes from K-folds estimation:
    def __print_outcomes(self):
        print('---------------------------------------------------------------------')
        print('\033[1mK-folds CV outcomes:\033[0m')
        print('Number of data folds: {0}.'.format(self.num_folds))

        if self.random_search:
            print('Number of samples for random search: {0}.'.format(self.n_samples))

        print('Estimation method: {0}.'.format(self.method.replace('_', ' ')))
        print('Metric for choosing best hyper-parameter: {0}.'.format(self.metric))
        print('Best hyper-parameters: {0}.'.format(self.best_param))

        if self.__using_default:
            print('Warning: given problems during K-folds CV estimation, hyper-parameters selected were those provided as default.')

        try:
            self.best_cv_metric = self.CV_metric[self.CV_metric['tun_param']==str(self.best_param)]['cv_' + self.metric].values[0]
            print('CV performance metric associated with best hyper-parameters: {0}.'.format(round(self.best_cv_metric, 4)))

        except:
            pass
        
        print('---------------------------------------------------------------------')

####################################################################################################################################
# K-folds CV estimation, refitting using all training data, and evaluating performance metrics on test data (when provided):

class Kfolds_fit(KfoldsCV):
    """
    Arguments for initialization: the same from KfoldsCV.
    
    Methods:
        "fit": runs K-folds CV with grid or random search, refits using all training data, and evaluate
        performance metrics on test data (when provided).
    
    Output objects: in addition to those from KfoldsCV:
        "model": final model fitted on the entire training data using the best values for hyper-parameters.
        "test_scores": dataframe with predicted values of response variable for test data.
        "cv_running_time": running time for K-folds CV estimation.
        "running_time": overall running time.
    """
    # Method that runs K-folds CV estimation, refits using all training data, and evaluate performance metrics on
    # test data (when provided):
    def fit(self, train_inputs, train_output, test_inputs=None, test_output=None, print_outcomes=True,
            print_time=True):
        """
        Method that runs K-folds CV estimation, refits using all training data, and evaluate performance metrics on
        test data (when provided).
        
        :param train_inputs: inputs of training data.
        :type train_inputs: dataframe.
        
        :param train_output: values for response variable of training data.
        :type train_output: dataframe.
        
        :param test_inputs: inputs of test data.
        :type test_inputs: dataframe.
        
        :param test_output: values for response variable of test data.
        :type test_output: dataframe.

        :param print_outcomes: defines whether final outcomes should be printed after execution.
        :type print_outcomes: boolean.

        :param print_time: defines whether total elapsed time should be printed after execution.
        :type print_time: boolean.
        """
        # Registering start time:
        start_time = datetime.now()
        
        # Dictionary for storing performance metrics:
        if test_inputs is not None:
            self.performance_metrics = {}
        
        # K-folds CV estimation:
        self.run(inputs=train_inputs, output=train_output, progress_bar=True, print_outcomes=False,
                 print_time=False)
        
        self.cv_running_time = self.running_time

        # Train-test estimation:
        if self.method == 'light_gbm':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'bagging_fraction': float(self.best_param['bagging_fraction']),
                     'learning_rate': float(self.best_param['learning_rate']),
                     'max_depth': int(self.best_param['max_depth']),
                     'num_iterations': int(self.best_param['num_iterations']),
                     'bagging_freq': 1,
                     'metric': self.best_param.get('metric') if self.best_param.get('metric') else '',
                     'verbose': -1}

            # Defining dataset for light GBM estimation:
            train_data = lgb.Dataset(data=train_inputs.values, label=train_output.values, params={'verbose': -1})

            # Training the model:
            self.model = lgb.train(params=param, train_set=train_data, num_boost_round=10, verbose_eval=False)

            # Predicting scores:
            if test_inputs is not None:
                self.test_scores = self.model.predict(test_inputs.values)

        elif self.method == 'xgboost':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'subsample': float(self.best_param['subsample']),
                     'eta': float(self.best_param['eta']),
                     'max_depth': int(self.best_param['max_depth'])}
            
            # Creating the training and validation data objects:
            dtrain = xgb.DMatrix(data=train_inputs, label=train_output)
            dtest = xgb.DMatrix(data=test_inputs)
            
            # Training the model:
            self.model = xgb.train(params=param, dtrain=dtrain,
                                   num_boost_round=int(self.best_param['num_boost_round']))
            
            # Predicting scores:
            if test_inputs is not None:
                self.test_scores = self.model.predict(dtest)
                
        else:
            # Creating estimation object:
            self.model = self._KfoldsCV__create_model(task=self.task, method=self.method, params=self.best_param)
            
            # Running estimation:
            self.model.fit(train_inputs, train_output)

            if test_inputs is not None:
                # Predicting scores:
                if self.task == 'classification':
                    self.test_scores = [p[1] for p in self.model.predict_proba(test_inputs)]

                else:
                    self.test_scores = [p for p in self.model.predict(test_inputs)]

        if test_inputs is not None:
            # Calculating performance metrics:
            self.__calculate_metrics(test_output=test_output)
            
            # Returning predicted scores:
            self.test_scores = pd.DataFrame(data={
                'test_score': self.test_scores,
                'y_true': test_output
            })
        
        # Registering end time:
        end_time = datetime.now()
        
        # Printing outcomes:
        if print_outcomes:
            self.__print_outcomes(test_inputs=test_inputs)
            print('\n')
        
        self.running_time = running_time(start_time=start_time, end_time=end_time, print_time=print_time)

    
    # Function that calculates performance metrics:
    def __calculate_metrics(self, test_output):
        if ('classification' in self.task) | ('binary' in self.task) | ('cross_entropy' in self.task):
            self.performance_metrics["test_roc_auc"] = roc_auc_score(test_output, self.test_scores)
            self.performance_metrics["test_prec_avg"] = average_precision_score(test_output, self.test_scores)
            self.performance_metrics["test_brier"] = brier_score_loss(test_output, self.test_scores)

        else:
            self.performance_metrics["test_rmse"] = np.sqrt(mean_squared_error(test_output, self.test_scores))
        
    # Function that prints outcomes from K-folds estimation:
    def __print_outcomes(self, test_inputs=None):
        print('---------------------------------------------------------------------')
        print('\033[1mTrain-test estimation outcomes:\033[0m')
        print('\n')
        
        print('Outcomes from K-folds CV estimation:')
        print('   Number of data folds: {0}.'.format(self.num_folds))

        if self.random_search:
            print('   Number of samples for random search: {0}.'.format(self.n_samples))

        print('   Estimation method: {0}.'.format(self.method.replace('_', ' ')))
        print('   Metric for choosing best hyper-parameter: {0}.'.format(self.metric))
        print('   Best hyper-parameters: {0}.'.format(self.best_param))

        if self._KfoldsCV__using_default:
            print('   Warning: given problems during K-folds CV estimation, hyper-parameters selected were those provided as default.')

        try:
            self.best_cv_metric = self.CV_metric[self.CV_metric['tun_param']==str(self.best_param)]['cv_' + self.metric].values[0]
            print('   CV performance metric associated with best hyper-parameters: {0}.'.format(round(self.best_cv_metric, 4)))

        except:
            pass
    
        if test_inputs is not None:
            print('\n')
            print('Performance metrics evaluated at test data:')
            for k in self.performance_metrics.keys():
                print('   {0} = {1}'.format(k, round(self.performance_metrics[k], 4)), sep='\t')
            print('---------------------------------------------------------------------')
