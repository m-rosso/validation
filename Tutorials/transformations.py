####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import text_clean

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Logarithm of selected features:

class log_transformation(object):
    """Applies function to log-transform all variables in a dataframe except for those
    explicitly declared. Returns the dataframe with selected variables log-transformed
    and their respective names changed to 'L#PREVIOUS_NAME()'."""

    def __init__(self, not_log):
        self.not_log = not_log
        
    def transform(self, data):
        # Function that applies natural logarithm to numerical variables:
        def log_func(x):
            """Since numerical features are not expected to assume negative values here, and since, after a sample
            assessment, only a few negative values were identified for just a few variables, suggesting the occurrence of
            technical issues for such observations, any negative values will be truncated to zero when performing
            log-transformation."""
            if x < 0:
                new_value = 0
            else:
                new_value = x

            transf_value = np.log(new_value + 0.0001)

            return transf_value
        
        # Redefining names of columns:
        new_col = []
        log_vars = []
        
        self.log_transformed = data
        
        # Applying logarithmic transformation to selected variables:
        for f in list(data.columns):
            if f in self.not_log:
                new_col.append(f)
            else:
                new_col.append('L#' + f)
                log_vars.append('L#' + f)
                self.log_transformed[f] = data[f].apply(log_func)

        self.log_transformed.columns = new_col
        
        print('\033[1mNumber of numerical variables log-transformed:\033[0m ' + str(len(log_vars)) + '.')

####################################################################################################################################
# Standardizing selected features:

class standard_scale(object):
    """Fits and transforms all variables in a dataframe, except for those explicitly defined to not scale.
    Uses 'StandardScaler' from sklearn and returns not only scaled data, but also in its dataframe original
    format. If test data is provided, then their values will be standardized using means and variances from
    train data."""
    
    def __init__(self, not_stand):
        self.not_stand = not_stand
    
    def scale(self, train, test=None):
        # Creating standardizing object:
        scaler = StandardScaler()
        
        # Calculating means and variances:
        scaler.fit(train.drop(self.not_stand, axis=1))
        
        # Standardizing selected variables:
        self.train_scaled = scaler.transform(train.drop(self.not_stand, axis=1))
        
        # Transforming data into dataframe and concatenating selected and non-selected variables:
        self.train_scaled = pd.DataFrame(data=self.train_scaled,
                                         columns=train.drop(self.not_stand, axis=1).columns)
        self.train_scaled.index = train.index
        self.train_scaled = pd.concat([train[self.not_stand], self.train_scaled], axis=1)
        
        # Test data:
        if test is not None:
            # # Standardizing selected variables:
            self.test_scaled = scaler.transform(test.drop(self.not_stand, axis=1))
            
            # Transforming data into dataframe and concatenating selected and non-selected variables:
            self.test_scaled = pd.DataFrame(data=self.test_scaled,
                                            columns=test.drop(self.not_stand, axis=1).columns)
            self.test_scaled.index = test.index
            self.test_scaled = pd.concat([test[self.not_stand], self.test_scaled], axis=1)

####################################################################################################################################
# Method that creates dummies from categorical features following a variance criterium for selecting categories:
class one_hot_encoding(object):
    """
    Arguments for initialization:
        'features': list of categorical features whose categories should be selected.
        'variance_param': parameter for selection based on the variance of a given dummy variable.
    Methods:
        'create_dummies': for a given training data ('categorical_train'), performs selection of dummies based on variance criterium.
        Then, creates the same set of dummy variables for test data ('categorical_test').
    Output objects:
        'self.categorical_features': list of categorical features whose categories should be selected.
        'self.variance_param': parameter for selection based on the variance of a given dummy variable.
        'self.dummies_train': dataframe with selected dummies for training data.
        'self.dummies_test': dataframe for test data with dummies selected from training data.
        'self.categories_assessment': dictionary with number of overall categories, number of selected categories, and selected
        categories for each categorical feature.
    """
    def __init__(self, categorical_features,  variance_param = 0.01):  
        self.categorical_features = categorical_features
        self.variance_param = variance_param

    def create_dummies(self, categorical_train, categorical_test = None):
        self.dummies_train = pd.DataFrame(data=[])
        self.dummies_test = pd.DataFrame(data=[])
        self.categories_assessment = {}
        
        # Loop over categorical features:
        for f in self.categorical_features:
            # Training data:
            # Creating dummy variables:
            dummies_cat = pd.get_dummies(categorical_train[f]) 
            dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

            # Selecting dummies_cat depending on their variance:
            selected_cat = [d for d in dummies_cat.columns if dummies_cat[d].var() > self.variance_param]

            # Dataframe with dummy variables for all categorical features (training data):
            self.dummies_train = pd.concat([self.dummies_train, dummies_cat[selected_cat]], axis=1)
            
            # Assessing categories:
            self.categories_assessment[f] = {
                "num_categories": len(dummies_cat.columns),
                "num_selected_categories": len(selected_cat),
                "selected_categories": selected_cat
            }

            if categorical_test is not None:
                # Test data:
                dummies_cat = pd.get_dummies(categorical_test[f])
                dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

                # Checking if all categories selected from training data also exist for test data:
                for c in selected_cat:
                    if c not in dummies_cat.columns:
                        dummies_cat[c] = [0 for i in range(len(dummies_cat))]

                # Dataframe with dummy variables for all categorical features (test data):
                self.dummies_test = pd.concat([self.dummies_test, dummies_cat[selected_cat]], axis=1)

                # Preserving columns order as the same for training data:
                self.dummies_test = self.dummies_test[list(self.dummies_train.columns)]

####################################################################################################################################
# Function that recriates original missing values from dummy variable of missing value status:
def recreate_missings(var, missing_var):
    """
    Arguments:
        'var': variable (series, array, or list) to impute missing values.
        'missing_var': variable (series, array, or list) that indicates missing data.
        Attention: both arguments should have the same lenght and should have the same index.
    Outputs:
        A list with missing values recreated (if any exists in 'missing_var').
    """
    var_list = list(var)
    missing_var_list = list(missing_var)
    new_values = []
    
    # Loop over observations:
    for i in range(len(var_list)):
        if missing_var_list[i] == 1:
            new_values.append(np.NaN)
        else:
            new_values.append(var_list[i])
    
    return new_values

####################################################################################################################################
# Function that treats missing values by imputing 0 whenever they are found:
def impute_missing(var):
    """
    Arguments:
        'var': variable (series, array, or list) whose missing values should be replaced by 0.
    Outputs:
        A dictionary containing a list of values for the variable after missing values treatment, and a list of
        missing value status.
    """
    var_list = list(var)
    new_values = []
    missing_var = []
    
    # Loop over observations:
    for value in var_list:
        if np.isnan(value):
            new_values.append(0)
            missing_var.append(1)
        else:
            new_values.append(value)
            missing_var.append(0)
    
    return {'var': new_values, 'missing_var': missing_var}

####################################################################################################################################
# Function that applies log-transformation:

def applying_log_transf(dataframe, not_log):
    """
    Function that applies log-transformation based upon the 'log_transformation' class.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :param not_log: list with features names that should not be log-transformed.
    :type not_log: list.

    :return: dataframe with numerical variables log-transformed.
    :rtype: dataframe.
    """
    log_dataframe = dataframe.copy()
    
    # Assessing missing values (before logarithmic transformation):
    num_miss = log_dataframe.isnull().sum().sum()
    
    # Applying the log-transformation:
    log_transf = log_transformation(not_log=not_log)
    log_transf.transform(log_dataframe)
    log_dataframe = log_transf.log_transformed

    # Assessing missing values (after logarithmic transformation):
    num_miss_log = log_dataframe.isnull().sum().sum()

    # Checking consistency in the number of missings:
    if num_miss_log != num_miss:
        print('Problem - Inconsistent number of overall missings!')
        
    # Assessing consistency of dimensions:
    if not log_dataframe.shape == dataframe.shape:
        print('Problem - Inconsistent dimensions!')
        print(f'Shape before scaling: {log_dataframe.shape}.\nShape after scaling: {dataframe.shape}.')
    
    return log_dataframe

####################################################################################################################################
# Function that applies the standard scaling transformation:

def applying_standard_scale(training_data, not_stand, *args, **kwargs):
    """
    Function that applies the standard scaling transformation based upon the 'standard_scale' class.

    The keyword arguments are expected to be dataframes whose argument names indicate the nature of the passed data. For instance,
    'test_data=df_test' would be a dataframe with test instances.

    :param training_data: reference data.
    :type training_data: dataframe.

    :param not_stand: list with features names that should not be standardized.
    :type not_stand: list.

    :return: dataframes with numerical variables standardized.
    :rtype: dictionary.
    """
    scaled_data = {}
    
    print('\033[1mStandard scaling training data...\033[0m')

    stand_scale = standard_scale(not_stand = not_stand)
    stand_scale.scale(train = training_data, test = None)

    scaled_data['training_data'] = stand_scale.train_scaled

    # Assessing consistency of dimensions:
    if not scaled_data['training_data'].shape == training_data.shape:
        print('Problem - Inconsistent dimensions!')
        print(f'Shape before scaling: {training_data.shape}.\nShape after scaling: {scaled_data["training_data"].shape}.')

    # Assessing consistency of missing values:
    num_miss = training_data.isnull().sum().sum()
    num_miss_scaled = scaled_data['training_data'].isnull().sum().sum()

    if num_miss_scaled != num_miss:
        print('Problem - Inconsistent number of overall missings!')

    # Loop over additional data:
    for d in kwargs.keys():
        print(f'\033[1mStandard scaling {d.replace("_", " ")}...\033[0m')
        stand_scale = standard_scale(not_stand = not_stand)
        stand_scale.scale(train = training_data, test = kwargs[d])

        scaled_data[d] = stand_scale.test_scaled

        # Assessing consistency of dimensions:
        if not scaled_data[d].shape == kwargs[d].shape:
            print('Problem - Inconsistent dimensions!')
            print(f'Shape before scaling: {training_data.shape}.\nShape after scaling: {scaled_data[d].shape}.')

        # Assessing consistency of missing values:
        num_miss = kwargs[d].isnull().sum().sum()
        num_miss_scaled = scaled_data[d].isnull().sum().sum()

        if num_miss != num_miss_scaled:
            print('Problem - Inconsistent number of overall missings!')
    
    return scaled_data

####################################################################################################################################
# Function that treats missing values:

def treating_missings(dataframe, cat_vars, drop_vars):
    """
    Function that treats missing values both from categorical and numerical features. The last set of features have their missings
    treated using the function 'impute_missing'.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :param cat_vars: list with categorical variables.
    :type cat_vars: list.

    :param drop_vars: list with support variables.
    :type drop_vars: list.

    :return: dataframe with treated missing values.
    :rtype: dataframe.
    """
    treated_dataframe = dataframe.copy()
    
    num_miss = treated_dataframe.isnull().sum().sum()
    
    # Loop over categorical features:
    num_miss_cat_treat = 0
    for f in cat_vars:
        treated_dataframe[f] = ['missing_value' if pd.isnull(x) else x for x in treated_dataframe[f]]
        num_miss_cat_treat += sum([1 for x in treated_dataframe[f] if x == 'missing_value'])
    
    # Loop over non-categorical features:
    for f in treated_dataframe.drop(drop_vars, axis=1).drop(cat_vars, axis=1):
        # Checking if there is missing values for a given feature:
        if treated_dataframe[f].isnull().sum() > 0:
            check_missing = impute_missing(treated_dataframe[f])
            treated_dataframe[f] = check_missing['var']
            treated_dataframe['NA#' + f.replace('L#', '')] = check_missing['missing_var']

    num_miss_treat = int(sum([sum(treated_dataframe[f]) for f in treated_dataframe.columns if 'NA#' in f]))
    num_miss_treat = num_miss_treat + num_miss_cat_treat

    if num_miss_treat != num_miss:
        print('Problem - Inconsistent number of overall missings!')
        print(f'Number of missings before treatment: {num_miss}.')
        print(f'Number of missings after treatment: {num_miss_treat}.')

    if treated_dataframe.isnull().sum().sum() > 0:
        print('Problem - Number of overall missings detected: ' +
              str(treated_dataframe.isnull().sum().sum()) + '.')
    
    return treated_dataframe

####################################################################################################################################
# Function that applies one-hot encoding transformation over categorical variables:

def applying_one_hot(training_data, cat_vars, variance_param=0.01, *args, **kwargs):
    """
    Function that applies one-hot encoding transformation over categorical variables based upon the 'one_hot_encoding' class.

    The keyword arguments are expected to be dataframes whose argument names indicate the nature of the passed data. For instance,
    'test_data=df_test' would be a dataframe with test instances.

    :param training_data: reference data.
    :type training_data: dataframe.
    
    :param cat_vars: list with categorical variables.
    :type cat_vars: list.

    :return: dataframes with categorical variables transformed into dummy variables.
    :rtype: dictionary.
    """
    dummies_df = {}
    transf_data = {}
    
    # Create object for one-hot encoding:
    categorical_transf = one_hot_encoding(categorical_features = cat_vars, variance_param = variance_param)

    # Treating texts:
    for f in cat_vars:
        training_data[f] = training_data[f].apply(text_clean)

    if kwargs:
        for d in kwargs:
            # Treating texts:
            for f in cat_vars:
                kwargs[d][f] = kwargs[d][f].apply(text_clean)

            # Creating dummies:
            categorical_transf.create_dummies(categorical_train = training_data[cat_vars],
                                              categorical_test = kwargs[d][cat_vars])

            # Additional data:
            dummies_df[d] = categorical_transf.dummies_test
            dummies_df[d].index = kwargs[d].index

            # Concatenating dummy variables with remaining columns and dropping out original categorical features:
            transf_data[d] = pd.concat([kwargs[d], dummies_df[d]], axis=1)
            transf_data[d].drop(cat_vars, axis=1, inplace=True)

    else:
        # Creating dummies:
        categorical_transf.create_dummies(categorical_train = training_data[cat_vars])
        
    # Training data:
    dummies_df['training_data'] = categorical_transf.dummies_train
    dummies_df['training_data'].index = training_data.index

    # Concatenating dummy variables with remaining columns and dropping out original categorical features:
    transf_data['training_data'] = pd.concat([training_data, dummies_df['training_data']], axis=1)
    transf_data['training_data'].drop(cat_vars, axis=1, inplace=True)
    
    print(f'\033[1mNumber of categorical features:\033[0m {len(cat_vars)}')
    print(f'\033[1mNumber of overall selected dummies:\033[0m {dummies_df["training_data"].shape[1]}.')
    
    return transf_data
