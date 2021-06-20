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

from concurrent.futures import ThreadPoolExecutor

from utils import running_time
from kfolds import KfoldsCV

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

"""
This module contains the class "BootstrapEstimation", which allows the implementation of bootstrap estimations. The core of this
procedure is to run a large collection of estimations, where each one of them is based on a different sample drawn from the
training data with the same length of it and considering replacement. An alternative to that, also available using this class, is to
produce samples without replacement (averaging), so estimations do not follow the bootstrap principle, but still permit the
calculation of statistics for performance metrics. This approach is particularly useful when data modeling is subject to high
variability, such as high-dimensional problems or the estimation of models based on decision trees (GBM, random forest, and so on).

The "BootstrapEstimation" class inherits from "KfoldsCV", even though the implementation of K-folds CV together with grid search or
random search is optional; just define "default_param" with desired values of hyper-parameters. Yet, it is more adequate to run
bootstrap estimations using K-folds CV for fine tuning at each iteration, since the hyper-parameters definition is a highly unstable
procedure. See documentation of "kfolds" module for more details on initialization, methods and attributes of "KfoldsCV" class,
which consequently apply for "BootstrapEstimation" as well.

The documentation within the "BootstrapEstimation" provides more detail over parameters of initialization other than those for the
"KfoldsCV" class. One that deserves special attention is "bootstrap_scores". When setting it to True, not only average and standard
deviation of performance metrics are calculated, but also the average of bootstrap predictions for each instance of test data is
defined and returned as an output attribute. Making use of them, additional performance metrics can be derived by opposing these
bootstrap scores against true labels.

For using this class, a train-test split is necessary, given that all performance metrics are evaluated on test data. The main
purpose of bootstrap estimations is to obtain a more reliable estimate of performance metrics in order to assess the impacts on
expected model performance of the introduction of new features, or any other intervention over the data modeling pipeline.

Note that bootstrap estimations are expected to be configured using a large number of iterations, like 100 or 1000 distinct runs. If
datasets are sufficienly large, averaging with less estimations (10-100 runs) may be a better option.
"""

####################################################################################################################################
# Bootstrap estimation with train-test split and K-folds CV for hyper-parameters definition:

class BootstrapEstimation(KfoldsCV):
    """
    Arguments for initialization: in addition to those from KfoldsCV:
        :param cv: defines whether K-folds CV should be ran in order to define the best values for
        hyper-parameters, either through grid search or random search.
        :type cv: boolean.
        
        :param replacement: defines whether to generate samples with replacement (bootstrap) or not (averaging).
        :type replacement: boolean.
        
        :param n_iterations: number of iterations of bootstrap or averaging estimation.
        :type n_iterations: integer (greater than zero).
        
        :param bootstrap_scores: defines whether to calculate bootstrap scores. Check documentation of the module
        for more information on bootstrap scores.
        :type bootstrap_scores: boolean.
    
    Methods:
        "run": method that runs bootstrap (averaging) estimation.
        "bootstrap_sampling": static method that produces random samples with replacement or not.
    
    Output objects:
    "performance_metrics": dictionary with performance metrics and best hyper-parameters for each one of the
    bootstrap iterations.
    "boot_stats": dictionary with average and standard deviation of performance metrics over bootstrap iterations.
    "most_freq_param": dictionary with the most frequently chosen hyper-parameters values.
    "boot_metrics": performance metrics calculated by averaging bootstrap predictions for each instance of test
    data, differently from the object "boot_stats" that follows from statistics for performance metrics that are
    calculated using individual bootstraps predictions.
    "boot_scores": averaged bootstrap predictions for each instance of test data.
    "running_time": overall running time.
    """         
    def __init__(self, task='classification', method='logistic_regression',
                 metric='roc_auc', num_folds=3, shuffle=False,
                 pre_selecting=False, pre_selecting_param=None,
                 random_search=False, n_samples=None,
                 grid_param=None, default_param=None,
                 parallelize=False,
                 cv=False, replacement=True, n_iterations=100, bootstrap_scores=False):
        KfoldsCV.__init__(self, task, method, metric, num_folds, shuffle, pre_selecting, pre_selecting_param,
                          random_search, n_samples, grid_param, default_param, parallelize)
        self.cv = cv
        self.replacement = replacement
        self.n_iterations = n_iterations
        self.bootstrap_scores = bootstrap_scores
    
    # Method that runs bootstrap (averaging) estimation:
    def run(self, train_inputs, train_output, test_inputs, test_output,
            val_inputs=None, val_output=None, print_outcomes=True, print_time=True):
        """
        Function that runs bootstrap (averaging) estimation.
        
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
        # Creating dictionary that receives performance metrics: 
        self.__create_metrics()
        
        # Registering start time:
        start_time = datetime.now()
        
        # Initializing progress bar for bootstrap n_iterations:
        bar = progressbar.ProgressBar(maxval=len(range(self.n_iterations)),
                                      widgets=[f'\033[1m{"Boostrap" if self.replacement else "Averaging"} estimation progress: \033[0m',
                                               progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()
        
        # Bootstrap scores:
        if self.bootstrap_scores:
            self.boot_scores = np.repeat(np.NaN, len(test_output))
        
        # Loop over bootstrap iterations:
        for r in range(self.n_iterations):
            # Creating bootstrap samples:
            boot_sample = self.bootstrap_sampling(inputs=train_inputs, output=train_output,
                                                  replacement=self.replacement)
            train_inputs_boot = boot_sample['inputs']
            train_output_boot = boot_sample['output']

            # Running K-folds CV estimation:
            if self.cv:
                KfoldsCV.run(self, inputs=train_inputs_boot, output=train_output_boot, progress_bar=False,
                             print_outcomes=False, print_time=False)
            
            else:
                self.best_param = self.default_param

            # Train-test estimation:
            if self.method in ['light_gbm', 'xgboost']:
                score_pred = self.__run_gbm(train_inputs_boot=train_inputs_boot, train_output_boot=train_output_boot,
                                            val_inputs=val_inputs, val_output=val_output, test_inputs=test_inputs)
            
            else:
                # Creating estimation object:
                model = self._KfoldsCV__create_model(task=self.task, method=self.method,
                                                          params=self.best_param)
                
                # Running estimation:
                model.fit(train_inputs_boot, train_output_boot)

                # Predicting scores:
                if self.task == 'classification':
                    score_pred = [p[1] for p in model.predict_proba(test_inputs)]

                else:
                    score_pred = [p for p in model.predict(test_inputs)]
            
            # Bootstrap estimation scores:
            if self.bootstrap_scores:
                self.boot_scores = [np.nansum([x, y]) for x,y in zip(self.boot_scores, score_pred)]
                                               
            # Calculating performance metrics:
            self.__calculate_metrics(test_output=test_output, score_pred=score_pred, step=r)
            
            # Bootstrap statistics for performance metrics:
            self.boot_stats = {}
            
            for k in [k for k in self.performance_metrics.keys() if k != 'best_param']:
                self.boot_stats[k.replace('test_', '')] = {
                    "mean": np.nanmean(self.performance_metrics[k]),
                    "std": np.nanstd(self.performance_metrics[k])
                }
            
            # Updating progress bar for bootstrap iterations:
            bar.update(list(range(self.n_iterations)).index(r)+1)
            time.sleep(0.1)
        
        # Bootstrap estimation scores:
        if self.bootstrap_scores:
            self.boot_scores = [s/(r+1) for s in self.boot_scores]
        
        # Registering end time:
        end_time = datetime.now()
        
        if print_outcomes:
            self.__print_outcomes()
        
        self.running_time = running_time(start_time=start_time, end_time=end_time, print_time=print_time)
    
    def __run_gbm(self, train_inputs_boot, train_output_boot, val_inputs, val_output, test_inputs):                              
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
            train_data = lgb.Dataset(data=train_inputs_boot.values, label=train_output_boot.values, params={'verbose': -1})

            if self.best_param.get('early_stopping_rounds'):
                val_data = lgb.Dataset(data=val_inputs.values, label=val_output.values, params={'verbose': -1})
                
                # Training the model:
                model = lgb.train(params=param, train_set=train_data, num_boost_round=10,
                                  valid_sets=[val_data], valid_names=['validation_data'],
                                  early_stopping_rounds=int(self.best_param['early_stopping_rounds']),
                                  verbose_eval=False)
                # Predicting scores:
                return model.predict(test_inputs.values, num_iteration=model.best_iteration)

            else:
                model = lgb.train(params=param, train_set=train_data, num_boost_round=10, verbose_eval=False)
                                               
                # Predicting scores:
                return model.predict(test_inputs.values)

        elif self.method == 'xgboost':
            # Create dictionary with parameters:
            param = {'objective': self.task,
                     'subsample': float(self.best_param['subsample']),
                     'eta': float(self.best_param['eta']),
                     'max_depth': int(self.best_param['max_depth']),
                     'eval_metric': self.best_param.get('eval_metric') if self.best_param.get('eval_metric') else None}

            # Creating the training and validation data objects:
            dtrain = xgb.DMatrix(data=train_inputs_boot, label=train_output_boot)
            dtest = xgb.DMatrix(data=test_inputs)
                                               
            # Training the model:
            if self.best_param.get('early_stopping_rounds'):
                dval = xgb.DMatrix(data=val_inputs, label=val_output)
                
                # Training the model:
                model = xgb.train(params=param, dtrain=dtrain,
                                  num_boost_round=int(self.best_param['num_boost_round']),
                                  evals=[(dval, 'val_data')],
                                  early_stopping_rounds=int(self.best_param['early_stopping_rounds']),
                                  verbose_eval=False)
                # Predicting scores:
                return model.predict(dtest, ntree_limit=model.best_iteration+1)

            else:
                model = xgb.train(params=param, dtrain=dtrain,
                                  num_boost_round=int(self.best_param['num_boost_round']))

                # Predicting scores:
                return model.predict(dtest)
                                               
    # Function that creates a dictionary that receives performance metrics: 
    def __create_metrics(self):
        if (self.task == 'classification') | (self.task == 'binary'):
            self.performance_metrics = {
                "test_roc_auc": [],
                "test_prec_avg": [],
                "test_brier": [],
                "best_param": []
            }
            
        else:
            self.performance_metrics = {
                "test_rmse": [],
                "best_param": []
            }                                
                                               
    # Function that calculates performance metrics:
    def __calculate_metrics(self, test_output, score_pred, step):
        if ('classification' in self.task) | ('binary' in self.task) | ('cross_entropy' in self.task):
            self.performance_metrics["test_roc_auc"].append(roc_auc_score(test_output, score_pred))
            self.performance_metrics["test_prec_avg"].append(average_precision_score(test_output, score_pred))
            self.performance_metrics["test_brier"].append(brier_score_loss(test_output, score_pred))

        else:
            self.performance_metrics["test_rmse"].append(np.sqrt(mean_squared_error(test_output, score_pred)))

        self.performance_metrics["best_param"].append(self.best_param)
        
        if self.bootstrap_scores:
            if (self.task == 'classification') | (self.task == 'binary'):
                self.boot_metrics = {
                    "roc_auc": roc_auc_score(test_output, [s/(step+1) for s in self.boot_scores]),
                    "prec_avg": average_precision_score(test_output, [s/(step+1) for s in self.boot_scores]),
                    "brier": brier_score_loss(test_output, [s/(step+1) for s in self.boot_scores])
                }

            else:
                self.boot_metrics = {
                    "rmse": np.sqrt(mean_squared_error(test_output, [s/(step+1) for s in self.boot_scores]))
                }
                                               
    # Function that prints outcomes from K-folds estimation:
    def __print_outcomes(self):
        print('---------------------------------------------------------------------------------------------')
        if self.replacement:
            print('\033[1mBootstrap statistics:\033[0m')
        else:
            print('\033[1mAveraging statistics:\033[0m')
        
        print('   Number of estimations: {}.'.format(self.n_iterations))
        for k in self.boot_stats.keys():
            print('   avg({0}) = {1}'.format(k, round(self.boot_stats[k]['mean'], 4)))
            print('   std({0}) = {1}'.format(k, round(self.boot_stats[k]['std'], 4)))
        
        if self.bootstrap_scores:
            print('\n')
            if self.replacement:
                print('\033[1m   Performance metrics based on bootstrap scores:\033[0m')
            else:
                print('\033[1m   Performance metrics based on averaging scores:\033[0m')
            
            for k in self.boot_metrics.keys():
                print('   {0} = {1}'.format(k, round(self.boot_metrics[k], 4)))
        
        if self.cv:
            print('\n')
            best_params = [str(p) for p in list(self.performance_metrics['best_param'])]
            best_params_freq = frequency_list(best_params)
            self.most_freq_param = list(best_params_freq.keys())[np.argmax(list(best_params_freq.values()))]
            print("   Most frequent best hyper-parameters: {0} ({1} out of {2} times).".format(self.most_freq_param,
                                                                                               np.nanmax(list(best_params_freq.values())),
                                                                                               self.n_iterations))
        else:
            print("   Hyper-parameters used in estimations: {}.".format(self.best_param))
        
        print('---------------------------------------------------------------------------------------------')
        print('\n')
        
    # Method that creates a bootstrap sample:
    @staticmethod
    def bootstrap_sampling(inputs, output, replacement = True):
        """
        Method that creates a bootstrap sample.
        
        :param inputs: data to be sampled (inputs).
        :type inputs: dataframe.
        
        :param output: data to be sampled (response variable).
        :type output: dataframe.
        
        :param replacement: defines whether samples should be taken with replacement (bootstrap) or not (averaging).
        :type replacement: boolean.
        
        :return: indexes of the sample.
        :rtype: dictionary.
        """
        n_sample = len(inputs)
        sample = sorted(np.random.choice(range(n_sample), size=n_sample, replace=replacement))

        return {'inputs': inputs.iloc[sample, :], 'output': output.iloc[sample]}
