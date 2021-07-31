####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import numpy as np
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error

from utils import cross_entropy_loss
from screening_features import VarScreeningNumerical, CorrScreeningNumerical

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

"""
This module contains a class that can implement three different classes of features selection methods: analytical methods,
supervised learning selection and exaustive methods.

Analytical methods refer to approaches where only two variables the most are taken at the same time: here, variance selection
(variance thresholding to exclude variables with too few variability) and correlation selection (correlation thresholding to exclude
variables with excessive correlation with variables with more variability) are explored by the class as initialization parameter
"method" is declared equal to "variance" or "correlation", respectively. Additional arguments needed during initialization:
    * threshold: reference value for variance or correlation thresholding.

Supervised learning selection means that only those features whose importance is greater than some threshold are selected, where
this importance is calculated as a model is trained under a given supervised learning method. Can be used as initialization
parameter "method" is defined equal to "supervised". Additional arguments needed:
    * estimator: (declared into "features_select" method) machine learning algorithm containing either a "coef_" or a
    "feature_importances_" attribute.
    * threshold: (declared during initialization) importance value above which features are selected.

Exaustive methods evaluate several distinct subsets of features with different lengths. Methods covered by the FeaturesSelection
class are RFE (method="rfe"), RFECV (method="rfecv"), SequentialFeatureSelector (method="sequential") and random selection
(method="random_selection"). Except for the last method, all the others are extracted from sklearn. Besides of a unified approach,
the FeaturesSelection class introduces the choice of the best number of features for RFE and SequentialFeatureSelector, since sklearn
implementation does not cover this.
    * RFE: given an initialized estimator, at each step a predefined number of the least relevant features are dropped until a given
    number is reached. Arguments (all declared during initialization, except for "estimator"):
        * estimator: machine learning algorithm.
        * num_folds: number of folds of K-folds CV for selecting final model.
        * metric: performance metric for selecting final model.
        * max_num_feats: maximum number of features to be tested.
        * step: number of features to be dropped at each iteration.
    
    * RFECV: selected features are defined according to the optimization of some performance metric that is calculated using K-folds
    cross-validation. Consequently, at each step the least important features are dropped, and from the final collection of models
    where each has a different number of features the best model is chosen through cross-validation. Arguments (all declared during
    initialization, except for "estimator"):
        * estimator: machine learning algorithm.
        * num_folds: number of folds of K-folds CV for selecting final model.
        * metric: performance metric for selecting final model.
        * min_num_feats: minimum number of features to be selected.
        * step: number of features to be dropped at each iteration.

    * SequentialFeatureSelector: depending on the "direction" initialization parameter, a version of forward-stepwise selection or a
    version of backward-stepwise selection can be implemented. As with RFE, the number of features to be selected is another parameter
    that should ultimately be defined in order to optimize model performance. Arguments (all declared during initialization, except
    for "estimator"):
        * estimator: machine learning algorithm.
        * num_folds: number of folds of K-folds CV for selecting final model.
        * metric: performance metric for selecting final model.
        * max_num_feats: maximum number of features to be tested.
        * direction: indicates whether forward or backward-stepwise selection should be implemented.

    * Random selection of features: defines a collection of models with different numbers of features (all randomly picked), and then
    chooses the best model using K-folds CV. Arguments (all declared during initialization, except for "estimator"):
        * estimator: machine learning algorithm.
        * num_folds: number of folds of K-folds CV for selecting final model.
        * metric: performance metric for selecting final model.
        * max_num_feats: maximum number of features to be tested.
        * step: number of features to be randomly included at each iteration.

The usage of the FeaturesSelection class requires the initialization of an object, when all relevant arguments should be declared
depending on the method chosen during the definition of "method" argument. Then, inputs and output are passed (together with
"estimator" parameter whenever needed) in the "select_features" method, which then produces a list with the names of selected
features.

Note in the FeaturesSelection class that "select_features" method is constructed upon three static methods that may be used as a
function without the need of creating an object from FeaturesSelection class. Then, inputs and output (together with "estimator"
parameter whenever needed) should also be declared when executing the static method.
"""

####################################################################################################################################
# Class that contains distinct methods of features selection:

class FeaturesSelection:
    """
    Arguments for initialization:
        :param method: defines which features selection method should be implemented.
        :type method: string.

        :param threshold: parameter of variance, correlation, and supervised learning selection. Respectively, consists of the
        variance below which features are disregarded, the correlation above which features are disregarded and the feature
        importance above which features are selected.
        :type threshold: float.

        :param num_folds: parameter of exaustive methods (RFE, RFECV, sequential selection, random selection). Given by the number of
        folds of K-folds CV for selecting final model.
        :type num_folds: integer.

        :param metric: parameter of exaustive methods (RFE, RFECV, sequential selection, random selection). Performance metric for
        selecting final model.
        :type metric: string.

        :param max_num_feats: parameter of exaustive methods (RFE, sequential selection, random selection). Maximum number of
        features to be tested.
        :type max_num_feats: integer.

        :param min_num_feats: parameter of exaustive methods (RFECV). Minimum number of features to be selected.
        :type min_num_feats: integer.

        :param step: parameter of exaustive methods (RFE, RFECV, random selection). Number of features to be dropped at each iteration
        (RFE, RFECV), or number of features to be randomly included at each iteration (random selection).
        :type step: integer.

        :param direction: parameter of exaustive methods (sequential selection). Indicates whether forward or backward-stepwise
        selection should be implemented.
        :type direction: string.

        :param seed: control random selection of features for reproducibility.
        :type seed: integer.

    Methods:
        "select_features": method that implements features selection according to the initialization parameters. Has inputs and output
        as main parameters, besides of the estimator for the following methods: supervised selection, RFE, RFECV, sequential selection,
        random selection.
        
        "analytical_selection": static method that implements analytical methods (variance and correlation selection).
        
        "supervised_selection": static method that implements supervised learning selection (selection of features based on their
        importances for training a model).

        "exaustive_selection": static method that implements one of the following exaustive methods: RFE, RFECV, sequential selection,
        random selection.

        "random_selector": static method that randomly selects features.

        "cv_performance": static method that calculates performance metrics based on K-folds cross-validation.

    Output objects:
        "selected_features": list with names of selected features.
    """
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
    
    def __init__(self, method, threshold=0, num_folds=5, metric=None,
                 max_num_feats=None, min_num_feats=None, step=1, direction='forward'):
        self.method = method
        
        if self.method in ['variance', 'correlation', 'supervised']:
            self.threshold = threshold
        
        if self.method in ['rfe', 'rfecv', 'sequential', 'random_selection']:
            self.num_folds = num_folds
            self.metric = metric
            self.max_num_feats = max_num_feats
            self.min_num_feats = min_num_feats
            self.step = step
            self.direction = direction
    
    def select_features(self, inputs, output=None, estimator=None):
        """
        Method that implements features selection according to the initialization parameters.

        :param inputs: input variables.
        :type inputs: dataframe.
        :param output: the outcome variable (not necessary for analytical methods).
        :type output: dataframe or series.
        :param estimator: learning method.
        :type estimator: object having fit and predict methods.
        """
        if self.method in ['variance', 'correlation']:
            self.selected_features = self.analytical_selection(inputs=inputs, method=self.method,
                                                               threshold=self.threshold)
            
        elif self.method=='supervised':
            self.selected_features = self.supervised_selection(estimator=estimator, inputs=inputs, output=output,
                                                               threshold=self.threshold)
        
        elif self.method in ['rfe', 'rfecv', 'sequential', 'random_selection']:
            self.selected_features = self.exaustive_selection(estimator=estimator, inputs=inputs, output=output,
                                                              method=self.method,
                                                              num_folds=self.num_folds, metric=self.metric,
                                                              max_num_feats=self.max_num_feats,
                                                              min_num_feats=self.min_num_feats, step=self.step,
                                                              direction=self.direction)
        else:
            return

    @staticmethod
    def analytical_selection(inputs, method='variance', threshold=0):
        """
        Static method that implements analytical methods (variance and correlation selection).

        :param inputs: input variables.
        :type inputs: dataframe.
        :param method: distinguishes between variance and correlation selection.
        :type method: string.
        :param threshold: variance below which or correlation above which features are disregarded.
        :type threshold: float.

        :return: names of selected features.
        :rtype: list.
        """
        # Features names:
        all_vars = list(inputs.columns)

        if method=='correlation':
            # Creating the object for correlation thresholding:
            analytical_method = CorrScreeningNumerical(features=all_vars, corr_threshold=threshold)

        else:
            # Creating the object for variance thresholding:
            analytical_method = VarScreeningNumerical(features=all_vars,
                                                      select_k=False, thresholding=True,
                                                      variance_threshold=threshold)

        # Selecting features based on their variances or correlation among them:
        analytical_method.select_feat(data=inputs)
        selected_features = analytical_method.selected_feat

        print(f'From {len(all_vars)} features, {len(selected_features)} were selected!')
        return selected_features
    
    @staticmethod
    def supervised_selection(estimator, inputs, output, threshold=0):
        """
        Static method that implements supervised learning selection (selection of features based on their importances for training a
        model).

        :param estimator: learning method.
        :type estimator: object having fit and predict methods.
        :param inputs: input variables.
        :type inputs: dataframe.
        :param output: the outcome variable.
        :type output: dataframe or series.
        :param threshold: value for the feature importance above which features are selected.
        :type: threshold: float.

        :return: names of selected features.
        :rtype: list.
        """
        # Features names:
        all_vars = list(inputs.columns)

        # Creating and training the model
        model = estimator.fit(X=inputs, y=output)

        # Defining the importance of each feature:
        feature_importances = model.coef_ if hasattr(model, 'coef_') else (model.feature_importances_ if hasattr(model, 'feature_importances_') else None)
        feature_importances = feature_importances.ravel()

        if feature_importances is not None:
            selected_features = [f for c, f in zip(feature_importances, inputs.columns) if abs(c)>threshold]
            
            print(f'From {len(all_vars)} features, {len(selected_features)} were selected!')
            return selected_features
        else:
            return
    
    @staticmethod
    def exaustive_selection(estimator, inputs, output, method='sequential', num_folds=5, metric=None,
                            max_num_feats=None, min_num_feats=None, step=1, direction='forward',
                            seed=None):
        """
        Static methods that implements one of the following exaustive methods.

        :param estimator: learning method.
        :type estimator: object having fit and predict methods.
        :param inputs: input variables.
        :type inputs: dataframe.
        :param output: the outcome variable.
        :type output: dataframe or series.
        :param method: distinguishes between RFE, RFECV, sequential or random selection.
        :type method: string.
        :param num_folds: number of folds of K-folds CV for selecting final model.
        :type num_folds: integer.
        :param metric: performance metric for selecting final model.
        :type metric: string.
        :param max_num_feats: parameter of exaustive methods (RFE, sequential selection, random selection). Maximum number of
        features to be tested.
        :type max_num_feats: integer.
        :param min_num_feats: minimum number of features to be selected (parameter for RFECV).
        :type min_num_feats: integer.
        :param step: number of features to be dropped at each iteration (RFE, RFECV), or number of features to be randomly included at
        each iteration (random selection).
        :type step: integer.
        :param direction: distinguishes if forward or backward-stepwise selection should be implemented.
        :type direction: string.
        :param seed: control random selection of features for reproducibility.
        :type seed: integer.

        :return: names of selected features.
        :rtype: list.
        """
        # Aliases of metrics:
        metric_sel = 'average_precision' if metric=='avg_precision_score' else metric
        
        selected_features = {}
        perf_metric = []

        # Features names:
        all_vars = list(inputs.columns)

        # Defining the maximum number of features to be selected:
        if max_num_feats is None: max_num_feats = len(all_vars)
        if max_num_feats > len(all_vars): max_num_feats = len(all_vars)

        if method in ['rfe', 'sequential']:
            # Loop over values for the number of features to be selected:
            for num_feats in range(1, max_num_feats+1):        
                if method=='rfe':
                    # Creating the object for the recursive feature elimination:
                    exaustive_method = RFE(estimator=estimator,
                                           n_features_to_select=num_feats,
                                           step=step)

                else:
                    # Avoiding the code to fail:
                    if num_feats > inputs.shape[1] - 1:
                        break

                    # Creating the object for the sequential feature selection:
                    exaustive_method = SequentialFeatureSelector(estimator=estimator,
                                                                 n_features_to_select=num_feats,
                                                                 direction=direction,
                                                                 scoring=metric_sel, cv=num_folds) 

                # Running the features selection:
                exaustive_method = exaustive_method.fit(X=inputs, y=output)

                # Selected features:
                selected_features[num_feats] = [c for s, c in zip(exaustive_method.support_, all_vars) if s]
                print(f'From {len(all_vars)} features, {len(selected_features[num_feats])} were selected!')

                # Calculating performance metric using K-folds cross-validation:
                perf_metric.append(FeaturesSelection.cv_performance(estimator=estimator,
                                                       inputs=inputs[selected_features[num_feats]],
                                                       output=output, num_folds=num_folds, metric=metric))

            # Best collection of features:
            if (metric == 'brier_loss') | (metric == 'mse'):
                best_collection = list(range(1, max_num_feats+1))[np.argmin(perf_metric)]

            else:
                best_collection = list(range(1, max_num_feats+1))[np.argmax(perf_metric)]

            print(f'\nFrom {len(all_vars)} features, {len(selected_features[best_collection])} were finally selected!')
            return selected_features[best_collection]
        
        elif method=='random_selection':
            # Loop over values for the number of features to be selected:
            for num_feats in range(step, max_num_feats+1, step):
                # Randomly selecting features:
                selected_features[num_feats] = FeaturesSelection.random_selector(features=all_vars,
                                                                                 num_feats=num_feats,
                                                                                 seed=seed)
                print(f'From {len(all_vars)} features, {len(selected_features[num_feats])} were selected!')

                # Calculating performance metric using K-folds cross-validation:
                perf_metric.append(FeaturesSelection.cv_performance(estimator=estimator,
                                                       inputs=inputs[selected_features[num_feats]],
                                                       output=output, num_folds=num_folds, metric=metric))
    
            # Best collection of features:
            if (metric == 'brier_loss') | (metric == 'mse'):
                best_collection = list(range(step, max_num_feats+1, step))[np.argmin(perf_metric)]

            else:
                best_collection = list(range(step, max_num_feats+1, step))[np.argmax(perf_metric)]

            print(f'\nFrom {len(all_vars)} features, {len(selected_features[best_collection])} were finally selected!')
            return selected_features[best_collection]
        
        else:
            # Creating the object for the recursive feature elimination with cross-validation:
            exaustive_method = RFECV(estimator=estimator,
                                     step=step,
                                     min_features_to_select=min_num_feats,
                                     cv=num_folds, scoring=metric_sel)
            
            # Running the features selection:
            exaustive_method = exaustive_method.fit(X=inputs, y=output)
            selected_features = [c for s, c in zip(exaustive_method.support_, all_vars) if s]
            
            print(f'From {len(all_vars)} features, {len(selected_features)} were selected!')
            return selected_features

    @staticmethod
    def random_selector(features, num_feats, seed=None):
        """
        Static method that implements random selection of features.

        :param features: list with names of features.
        :type features: list or array.
        :param num_feats: number of features to be randomly selected.
        :type num_feats: integer.
        :param seed: control random selection of features for reproducibility.
        :type seed: integer.

        :return: list of features randomly selected from a complete collection.
        :rtype: list.
        """
        np.random.seed(seed=seed)
        
        return list(np.random.choice(features, num_feats, replace=False))
        
    @staticmethod
    def cv_performance(estimator, inputs, output, num_folds, metric):
        """
        Static method that calculates performance metrics based on K-folds cross-validation.

        :param estimator: learning method.
        :type estimator: object having fit and predict methods.
        :param inputs: input variables.
        :type inputs: dataframe.
        :param output: the outcome variable.
        :type output: dataframe or series.
        :param num_folds: number of folds of K-folds CV.
        :type num_folds: integer.
        :param metric: performance metric.
        :type metric: string.

        :return: performance metric evaluated through K-folds cross-validation.
        :rtype: float.
        """
        performance_metric = []

        # Loop over folds of data:
        for train, test in KFold(num_folds).split(inputs):
            # Creating the model:
            model = estimator.fit(inputs.iloc[train, :], output.iloc[train])

            # Defining validation data:
            X_test = inputs.iloc[test, :]
            y_test = output.iloc[test]

            # Prediction on held-out data:
            predictions = model.predict(X_test)

            # K-folds CV estimate of performance metric:
            performance_metric.append(FeaturesSelection.__metrics_functions[metric](y_test, predictions))

        return np.nanmean(performance_metric)
