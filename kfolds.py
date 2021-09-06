####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'
__repo__ = 'https://github.com/m-rosso/validation'

import pandas as pd
import numpy as np
import json
import os

from datetime import datetime
import time
import progressbar
from concurrent.futures import ThreadPoolExecutor
from itertools import product

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error

# pip install lightgbm
import lightgbm as lgb
# pip install xgboost
import xgboost as xgb

from utils import running_time, cross_entropy_loss
from features_selection import FeaturesSelection

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

KfoldsCV and all classes that inherit from it allow a large amount of control over hyper-parameters. There are three initialization
parameters relevant for this: "grid_param", a dictionary with hyper-parameters as keys and corresponding values to be explored;
"default_param", a dictionary with values of hyper-parameters to be used when some possibilities fail to return valid results
during grid/random search; and "fixed_params", hyper-parameters that should not be explored during optimization, but instead of
using libraries default values, it is possible to choose them appropriately as a function of the application.

Setting "random_search" to False and declaring "grid_param", one can execute grid search, which is more likely to provide fine results
when previous studies have been done so only most promising alternatives remain left. Setting "random_search" to True and declaring
"grid_param" as a dictionary whose keys are the most relevant hyper-parameters of a ML method and whose respective values are lists
with integer values or statistical distributions from scipy.stats (such as "uniform", "norm", "randint"), then random search can be
implemented. This alternative is specially suited for ML methods that have multiple hyper-parameters. Finally, when random_search is
set to True, "n_samples" should be declared for the number of randomly picked combinations of hyper-parameters.

"task" and "method" are the two fundamental parameters for initializing those classes. Supervised learning tasks available using these
classes are binary classification and regression. Some slight modifications are required for implementing multiclass classification
(which should be done soon). For these two problems, the following ML methods are supported:
	1) Logistic regression (from sklearn).
		* Main hyper-parameters for tuning: regularization parameter ('C').
	2) Linear regression (Lasso) (from sklearn).
		* Main hyper-parameters for tuning: regularization parameter ('C').
	3) GBM (sklearn).
		* Main hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('learning_rate'),
        number of estimators ('n_estimators').
    4) GBM (LightGBM).
        * Main hyper-parameters for tuning: subsample ('bagging_fraction'), maximum depth ('max_depth'), learning rate ('learning_rate'),
        number of estimators ('num_iterations').
        * By declaring 'metric' and 'early_stopping_rounds' into the parameters dictionary, it is possible to implement both "KfoldsCV"
        and "Kfolds_fit" with early stopping. For "KfoldsCV", at each k-folds estimation early stopping will take place, while for
        "Kfolds_fit" estimation will stop after a stopping rule is triggered both during each of k-folds estimation and during the
        final fitting using the entire training data.
    5) GBM (XGBoost).
        * Main hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('eta'), number of
        estimatores ('num_boost_round').
        * By declaring 'eval_metric' and 'early_stopping_rounds' into the parameters dictionary, also for XGBoost early stopping is
        available for both "KfoldsCV" and "Kfolds_fit".
	6) Random forest (from sklearn).
		* Main hyper-parameters for tuning: number of estimators ('n_estimators'), maximum number of features ('max_features') and minimum
        number of samples for split ('min_samples_split').
	7) SVM (from sklearn).
		* Main hyper-parameters for tuning: regularization parameter ('C') kernel ('kernel'), polynomial degree ('degree'), gamma
        ('gamma').

Performance metrics allowed for binary classification are ROC-AUC, average precision score (as proxy for precision-recall AUC), and
Brier score. For regression, RMSE, MAE, R2 and log-MSE are the metrics available. These metrics are used for optimizing
hyper-parameters and also for reference in terms of evaluating a trained model.

Declaring "parallelize" equal to True when initializing the K-folds CV object is expected to improve running time, since
training-validation estimation for all K folds of data is then implemented in parallel.

KfoldsCV and classes that inherit from it cover features selection. First, one may choose among three different classes of
features selection methods: analytical methods (variance and correlation thresholding), supervised learning selection (picking
features according to feature importances as provided by supervised learning methods) and exaustive methods (RFE, RFECV,
sequential selection, random selection). Second, when using "KfoldsCV_fit" class, one may choose between applying features
selection at each iteration of K-folds cross-validation or only when final model is trained. These features selection tools
derived from "FeaturesSelection" class, which in its turn follows from some sklearn classes and some independently developed
classes and functions.

Recognizing that sklearn already offers its own classes and functions for cross-validation, grid search, and random search, these
are the main features that may supplement the use of sklearn:
	1) Simplified usage: its interface is maybe more intuitive, since it requires only the creation of an object from "KfoldsCV" or
	from "KfoldsCV_fit". All relevant parameters are then initialized, and only a method for running/fitting should be executed
	having as arguments only the data. Finally, all relevant outcomes from executing these validation methods are kept together with
	the object as attributes.

	2) Additional features: more information is provided by the end of estimations, such as cross-validation predictions for
	training data, performance metrics, progress bar and elapsed time.

	3) Pre-selection of features: pre-selecting features inside K-folds CV allows to run random/grid search without the fear of
    incurring in biases from selecting features in a first place and then proceeding to other supervised learning tasks. When it
    comes to pre-selection of features, all these classes are very flexibly, both in methods available and in terms of it usage.

	4) More flexibility: by changing components of method "__create_model", models from any library can be applied, not only those
	provided by sklearn, all in the same framework. Currently, LightGBM and XGBoost are available, but also neural networks from Keras should
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
        
        :param pre_selecting_params: parameters for a given features selection method.
        :type pre_selecting_params: dictionary.
        
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
    
        :param fixed_params: values for hyper-parameters that should be changed from default but not optimized.
        :type fixed_params: dictionary (strings as keys and floats/integers/strings as values).
    
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
        "best_param": dictionary with the best value for each hyper-parameter.
        "running_time": overall running time.
    """
    
    # Dictionary with functions for computing performance metrics:
    __metrics_functions = {
        'roc_auc': roc_auc_score,
        'avg_precision_score': average_precision_score,
        'brier_loss': brier_score_loss,
        'mse': mean_squared_error,
        'cross_entropy': cross_entropy_loss,
        'r2': r2_score,
        'msle': mean_squared_log_error,
        'mae': mean_absolute_error
    }
    
    def __init__(self, task='classification', method='logistic_regression',
                 metric='roc_auc', num_folds=3, shuffle=False,
                 pre_selecting=False, pre_selecting_params=None,
                 random_search=False, n_samples=None,
                 grid_param=None, default_param=None, fixed_params=None,
                 parallelize=False):
        self.task = task
        self.method = str(method)
        self.metric = metric
        self.num_folds = int(num_folds)
        self.shuffle = shuffle
        self.default_param = default_param
        self.fixed_params = fixed_params
        self.pre_selecting = pre_selecting
        self.pre_selecting_params = pre_selecting_params
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
            grid_param = []
            
            # Loop over combinations of hyper-parameters:
            for i in eval('product(' + ','.join([str(params[p]) for p in params]) + ')'):
                grid_param.append(dict(zip(params.keys(), list(i))))

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

        # Pre-selection of features:
        if self.pre_selecting:
            selected_features = self.__pre_selection(input_train=X_train, output_train=y_train,
                                                     pre_selecting_params=self.pre_selecting_params)

            X_train = X_train[selected_features]
            X_val = X_val[selected_features]

        if self.method in ['light_gbm', 'xgboost']:
            score_pred = self.__run_gbm(train_inputs=X_train, train_output=y_train, test_inputs=X_val,
                                        test_output=y_val, param_idx=param_idx)

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
    
    # Function that creates, trains and returns predictions for LightGBM and XGBoost models:
    def __run_gbm(self, train_inputs, train_output, test_inputs, test_output, param_idx):
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

            # Defining dataset for LightGBM estimation:
            train_data = lgb.Dataset(data=train_inputs.values, label=train_output.values, params={'verbose': -1})

            # Creating and training the model:
            if self.grid_param[param_idx].get('early_stopping_rounds'):
                val_data = lgb.Dataset(data=test_inputs.values, label=test_output.values, params={'verbose': -1})
                model = lgb.train(params=param, train_set=train_data, num_boost_round=10,
                                  valid_sets=[val_data], valid_names=['validation_data'],
                                  early_stopping_rounds=int(self.grid_param[param_idx]['early_stopping_rounds']),
                                  verbose_eval=False)
                
                # Predicting scores:
                return model.predict(test_inputs.values, num_iteration=model.best_iteration)
                
            else:
                model = lgb.train(params=param, train_set=train_data, num_boost_round=10, verbose_eval = False)

                # Predicting scores:
                return model.predict(test_inputs.values)

        elif self.method == 'xgboost':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'subsample': float(self.grid_param[param_idx]['subsample']),
                     'eta': float(self.grid_param[param_idx]['eta']),
                     'max_depth': int(self.grid_param[param_idx]['max_depth']),
                     'eval_metric': self.grid_param[param_idx].get('eval_metric') if self.grid_param[param_idx].get('eval_metric') else None,
                    }

            # Creating the training and validation data objects:
            dtrain = xgb.DMatrix(data=train_inputs, label=train_output)
            dval = xgb.DMatrix(data=test_inputs, label=test_output)

            # Training the model:
            if self.grid_param[param_idx].get('early_stopping_rounds'):
                model = xgb.train(params=param, dtrain=dtrain,
                                  num_boost_round=int(self.grid_param[param_idx]['num_boost_round']),
                                  evals=[(dval, 'val_data')],
                                  early_stopping_rounds=int(self.grid_param[param_idx]['early_stopping_rounds']),
                                  verbose_eval=False)
                
                # Predicting scores:
                return model.predict(dval, ntree_limit=model.best_iteration+1)

            else:
                model = xgb.train(params=param, dtrain=dtrain, num_boost_round=int(self.grid_param[param_idx]['num_boost_round']))

                # Predicting scores:
                return model.predict(dval)
    
    # Function that applies features selection:
    def __pre_selection(self, input_train, output_train, pre_selecting_params):
        """
        Executes a features selection method which finally selects a subset of features to be kept for model
        estimation.
        
        :param input_train: inputs of training data.
        :type input_train: dataframe.

        :param output_train: values for response variable of training data.
        :type output_train: dataframe.
        
        :param pre_selecting_params: parameters for a given features selection method.
        :type pre_selecting_params: dictionary.
        
        :return: list of pre-selected features.
        :rtype: list.
        """
        # Creating the object for features selection:
        selection = FeaturesSelection(method=pre_selecting_params.get('method'),
                                      threshold=pre_selecting_params.get('threshold'),
                                      num_folds=pre_selecting_params.get('num_folds'),
                                      metric=pre_selecting_params.get('metric'),
                                      max_num_feats=pre_selecting_params.get('max_num_feats'),
                                      min_num_feats=pre_selecting_params.get('min_num_feats'),
                                      step=pre_selecting_params.get('step'),
                                      direction=pre_selecting_params.get('direction'))

        # Running the features selection:
        selection.select_features(inputs=input_train,
                                  output=output_train,
                                  estimator=pre_selecting_params.get('estimator'))
        
        return selection.selected_features

    # Function that creates estimation objects:
    def __create_model(self, task, method, params):
        if task == 'classification':
            # Translating inserted method into its sklearn name:
            methods_dict = {'logistic_regression': 'LogisticRegression', 'GBM': 'GradientBoostingClassifier',
                            'random_forest': 'RandomForestClassifier', 'SVM': 'SVC'}
            
            # Constructing the model:
            model = self.__construct_model(method=methods_dict[method], params=params)


        elif task == 'regression':
            # Translating inserted method into its sklearn name:
            methods_dict = {'lasso': 'Lasso', 'GBM': 'GradientBoostingRegressor',
                            'random_forest': 'RandomForestRegressor', 'SVM': 'SVR'}
            
            # Constructing the model:
            model = self.__construct_model(method=methods_dict[method], params=params)

        return model

    # Function that initializes models from declared method and parameters:
    def __construct_model(self, method, params):
        # Dictionary with all parameters:
        complete_params = {}
        complete_params.update(params)
        
        # Including fixed parameters declared during the initialization of the class:
        if self.fixed_params is not None:
            complete_params.update(self.fixed_params)

        # Handling data types:
        for p in complete_params.keys():
            complete_params[p] = f'\'{complete_params[p]}\'' if isinstance(complete_params[p], str) else complete_params[p]
            
        # Concatenating all parameters:
        init_params = ', '.join([f'{p}={complete_params[p]}' for p in complete_params.keys()])

        # Constructing the model from the method and initialization parameters:
        return eval(f'{method}({init_params})')
    
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
    Arguments for initialization: in addition to those from KfoldsCV:
        :param only_final_selection: defines whether features selection should occur only in the estimation of
        the final model.
        :type only_final_selection: boolean.
    
    Methods:
        "fit": runs K-folds CV with grid or random search, refits using all training data, and evaluate
        performance metrics on test data (when provided).
    
    Output objects: in addition to those from KfoldsCV:
        "model": final model fitted on the entire training data using the best values for hyper-parameters.
        "test_scores": dataframe with predicted values of response variable for test data.
        "cv_running_time": running time for K-folds CV estimation.
        "running_time": overall running time.
    """
    def __init__(self, task='classification', method='logistic_regression',
                 metric='roc_auc', num_folds=3, shuffle=False,
                 pre_selecting=False, pre_selecting_params=None, only_final_selection=False,
                 random_search=False, n_samples=None,
                 grid_param=None, default_param=None, fixed_params=None,
                 parallelize=False,
                ):
        KfoldsCV.__init__(self, task, method, metric, num_folds, shuffle, pre_selecting, pre_selecting_params,
                          random_search, n_samples, grid_param, default_param, fixed_params, parallelize)
        self.only_final_selection = only_final_selection
    
    # Method that runs K-folds CV estimation, refits using all training data, and evaluate performance metrics on
    # test data (when provided):
    def fit(self, train_inputs, train_output,
            val_inputs=None, val_output=None,
            test_inputs=None, test_output=None,
            progress_bar=True, print_outcomes=True, print_time=True):
        """
        Method that runs K-folds CV estimation, refits using all training data, and evaluate performance metrics on
        test data (when provided).
        
        :param train_inputs: inputs of training data.
        :type train_inputs: dataframe.
        
        :param train_output: values for response variable of training data.
        :type train_output: dataframe.
        
        :param val_inputs: inputs of validation data (used for early stopping with LightGBM or XGBoost).
        :type val_inputs: dataframe.
        
        :param val_output: values for response variable of validation data (used for early stopping with LightGBM or XGBoost).
        :type val_output: dataframe.

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
        self.pre_selecting = False if self.only_final_selection else self.pre_selecting
        
        self.run(inputs=train_inputs, output=train_output, progress_bar=progress_bar, print_outcomes=False,
                 print_time=False)
        
        self.cv_running_time = self.running_time

        # Pre-selection of features:
        if (self.pre_selecting) | (self.only_final_selection):
            selected_features = self._KfoldsCV__pre_selection(input_train=train_inputs, output_train=train_output,
                                                              pre_selecting_params=self.pre_selecting_params)
            self.num_selected_features = len(selected_features)
            
            train_inputs = train_inputs[selected_features]
            
            if test_inputs is not None:
                test_inputs = test_inputs[selected_features]

        # Train-test estimation:
        if self.method in ['light_gbm', 'xgboost']:
            self.__run_gbm(train_inputs=train_inputs, train_output=train_output,
                           val_inputs=val_inputs, val_output=val_output,
                           test_inputs=test_inputs)
                
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

    def __run_gbm(self, train_inputs, train_output, val_inputs, val_output, test_inputs):
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

            # Defining dataset for LightGBM estimation:
            train_data = lgb.Dataset(data=train_inputs.values, label=train_output.values, params={'verbose': -1})

            if self.best_param.get('early_stopping_rounds'):
                val_data = lgb.Dataset(data=val_inputs.values, label=val_output.values, params={'verbose': -1})
                
                # Training the model:
                self.model = lgb.train(params=param, train_set=train_data, num_boost_round=10,
                                       valid_sets=[val_data], valid_names=['validation_data'],
                                       early_stopping_rounds=int(self.best_param['early_stopping_rounds']),
                                       verbose_eval=False)
                self.best_iteration = self.model.best_iteration

                # Predicting scores:
                if test_inputs is not None:
                    self.test_scores = self.model.predict(test_inputs.values, num_iteration=self.model.best_iteration)

            else:
                self.model = lgb.train(params=param, train_set=train_data, num_boost_round=10, verbose_eval=False)

                # Predicting scores:
                if test_inputs is not None:
                    self.test_scores = self.model.predict(test_inputs.values)

        elif self.method == 'xgboost':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'subsample': float(self.best_param['subsample']),
                     'eta': float(self.best_param['eta']),
                     'max_depth': int(self.best_param['max_depth']),
                     'eval_metric': self.best_param.get('eval_metric') if self.best_param.get('eval_metric') else None}

            # Creating the training and validation data objects:
            dtrain = xgb.DMatrix(data=train_inputs, label=train_output)
            dtest = xgb.DMatrix(data=test_inputs)

            # Training the model:
            if self.best_param.get('early_stopping_rounds'):
                dval = xgb.DMatrix(data=val_inputs, label=val_output)
                
                # Training the model:
                self.model = xgb.train(params=param, dtrain=dtrain,
                                       num_boost_round=int(self.best_param['num_boost_round']),
                                       evals=[(dval, 'val_data')],
                                       early_stopping_rounds=int(self.best_param['early_stopping_rounds']),
                                       verbose_eval=False)
                self.best_iteration = self.model.best_iteration
                
                # Predicting scores:
                if test_inputs is not None:
                    self.test_scores = self.model.predict(dtest, ntree_limit=self.model.best_iteration+1)

            else:
                self.model = xgb.train(params=param, dtrain=dtrain,
                                       num_boost_round=int(self.best_param['num_boost_round']))

                # Predicting scores:
                if test_inputs is not None:
                    self.test_scores = self.model.predict(dtest)
    
    # Function that calculates performance metrics:
    def __calculate_metrics(self, test_output):
        if ('classification' in self.task) | ('binary' in self.task) | ('cross_entropy' in self.task):
            self.performance_metrics["test_roc_auc"] = roc_auc_score(test_output, self.test_scores)
            self.performance_metrics["test_prec_avg"] = average_precision_score(test_output, self.test_scores)
            self.performance_metrics["test_brier"] = brier_score_loss(test_output, self.test_scores)

        else:
            self.performance_metrics["test_rmse"] = np.sqrt(mean_squared_error(test_output, self.test_scores))
            self.performance_metrics["test_r2"] = r2_score(test_output, self.test_scores)
            self.performance_metrics["test_mae"] = mean_absolute_error(test_output, self.test_scores)
            
            try:
                self.performance_metrics["test_msle"] = mean_squared_log_error(test_output, self.test_scores)
            except ValueError as error:
                self.performance_metrics["test_msle"] = np.NaN
        
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
