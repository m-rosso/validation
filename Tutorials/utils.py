####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

__author__ = 'm-rosso'

import pandas as pd
import numpy as np

from datetime import datetime
import time

import re
# pip install unidecode
from unidecode import unidecode

from statsmodels.stats.proportion import proportions_ztest

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function that converts epoch into date:

def epoch_to_date(x):
    if np.isnan(x):
        return np.NaN

    else:
        str_datetime = time.strftime('%d %b %Y %H:%M:%S', time.localtime(x/1000))
        dt = datetime.strptime(str_datetime, '%d %b %Y %H:%M:%S')
        return dt

####################################################################################################################################
# Function that returns the size of each store:

def store_size(dataframe, size_var = 'last_approved', criteria = {'pequena': 500, 'grande': 2500}):
    """
    Function that indicates the size of store among small, medium, and large.
    
    Arguments:
        'dataframe': dataframe containing the data for classification.
        'size_var': string with the name of the variable for classification.
        'criteria': dictionary with criteria for classifying a store between 'pequena' and 'grande' with remaining
        stores corresponding to 'media'.
    
    Output:
        Returns a list with classes according with the stores sizes ('pequena', 'media', 'grande').
    """
    
    sizes = []

    # Loop over data points:
    for i in range(len(dataframe)):
        if dataframe.iloc[i][size_var] <= criteria['pequena']:
            sizes.append('pequena')

        elif (dataframe.iloc[i][size_var] > criteria['pequena']) & (dataframe.iloc[i][size_var] <= criteria['grande']):
            sizes.append('media')

        else:
            sizes.append('grande')

    return sizes

####################################################################################################################################
# Function that produces a random sample preserving classes distribution of a categorical variable:

def representative_sample(dataframe, categorical_var, classes, sample_share=0.5):
    """
    Arguments:
        'dataframe': dataframe containing indices to be drawn and a categorical variable whose distribution in
        the sample should be kept equal to that of whole data.
        'categorical_var': categorical variable of reference (string).
        'classes': dictionary whose keys are classes and values are their shares in the entire data.
        'sample_share': float indicating sample length as the proportion of entire data length.
    Output:
        Returns a list with randomly picked indices.
    """
    
    # Randomly picked indices:
    samples = [sorted(np.random.choice(dataframe[dataframe[categorical_var]==k].index,
                                       size=int(classes[k]*sample_share*len(dataframe)),
                                       replace=False)) for k in classes.keys()]
    
    sample = []

    # Loop over samples:
    for l in samples:
        # Loop over indices:
        for i in l:
            sample.append(i)
    
    return sample

####################################################################################################################################
# Function that converts status_id into status:

def convert_status(id_var):
    """
    Function that converts status_id into status.
    
    Arguments:
        'id_var': integer (or string), possible values within [0, 1, 2, 3, 4, 5, 90].
    
    Outputs:
        Returns a string with the status name.
    """
    
    # Dictionary with status for each id:
    status_dict = {
        '0': 'pending',
        '1': 'approved',
        '2': 'declined',
        '3': 'fraud',
        '4': 'not_authorized',
        '5': 'canceled',
        '90': 'not_analyzed'
    }
    
    return status_dict[str(id_var)]

####################################################################################################################################
# Function that identifies if a given feature name corresponds to a velocity:

def is_velocity(string):
    if ('C#' in string) | ('NA#' in string):
        return False
    
    x1 = string.split('(')
    
    if len(x1) <= 1:
        return False

    x2 = x1[1]       

    if len(x2) <= 1:
        return False
    
    check = 0
    x3 = x2.split(')')[0].split(',')
    
    if len(x3) == 2:
        first_clause = len([1 for d in '0123456789' if d in x3[0]]) == 0
        second_clause = len([1 for d in '0123456789' if d in x3[1]]) > 0
        third_clause = len([1 for l in 'abcdefghijklmnopqrstuvxwyzç' if l in str.lower(x3[0])]) > 0
        fourth_clause = len([1 for l in 'abcdefghijklmnopqrstuvxwyzç' if l in str.lower(x3[1])]) == 0
        
        if first_clause & second_clause & third_clause & fourth_clause:
            check += 1
    
    return check > 0

####################################################################################################################################
# Function for cleaning texts:

def text_clean(text, lower=True):
    if pd.isnull(text):
        return text
    
    else:
        text = str(text)

        # Removing accent:
        text_cleaned = unidecode(text)
        # try:
        #     text_cleaned = unidecode(text)
        # except AttributeError as error:
        #     print(f'Error: {error}.')

        # Removing extra spaces:
        text_cleaned = re.sub(' +', ' ', text_cleaned)
        
        # Removing spaces before and after text:
        text_cleaned = str.strip(text_cleaned)
        
        # Replacing spaces:
        text_cleaned = text_cleaned.replace(' ', '_')
        
        # Deleting signs:
        for m in '.,;+-!@#$%¨&*()[]{}\\/|':
            if m in text_cleaned:
                text_cleaned = text_cleaned.replace(m, '')

        # Setting text to lower case:
        if lower:
            text_cleaned = text_cleaned.lower()

        return text_cleaned

####################################################################################################################################
# Function that creates a residual for non-selected categories:

def create_residual(values, selected_categories):
    """
    Function that creates a residual for non-selected categories.
    
    Arguments:
        'values': list, series, or array with original categories for a collection of instances.
        'selected_categories': list of predefined categories.
        
    Output:
        Returns a list with final categories.
    """
    
    new_values = []
    
    for i in values:
        if i not in selected_categories:
            new_values.append('residual_category')
        
        else:
            new_values.append(i)
    
    return new_values

####################################################################################################################################
# Function that loads data:

def loading_data(path, dtype, id_var, print_info=True):
    """
    Function that loads dataframes from '.csv' files.
    
    :param path: path to the file that should be imported.
    :type path: string.

    :param dtype: specifies data types that should be forced 
    :type dtype: dictionary.

    :param id_var: variable that identifies unique instances.
    :type id_var: string.

    :param print_info: variable that indicates whether information about imported dataset should be printed.
    :type print_info: boolean.

    :return: dataframe constructed upon the imported file.
    :rtype: dataframe.
    """
    # Importing data:
    dataframe = pd.read_csv(path, dtype=dtype)
    
    # Converting epoch into datetime:
    dataframe['date'] = dataframe['epoch'].apply(lambda x: epoch_to_date(float(x)))

    if print_info:
        print(f'Shape of df: {dataframe.shape}.')
        print(f'Number of distinct instances: {dataframe.order_id.nunique()}.')
        print('Time period: from {0} to {1}.'.format(str(dataframe.date.apply(lambda x: x.date()).min()),
                                                     str(dataframe.date.apply(lambda x: x.date()).max())))

    # Checking duplicated data:
    if len(dataframe) != dataframe[id_var].nunique():
        print('Problem - There are duplicated instances!')
    
    return dataframe

####################################################################################################################################
# Function that classifies columns from a dataframe:

def classify_features(dataframe, drop_vars=[], drop_excessive_miss=True, drop_no_var=True,
                      validation_data=None, test_data=None):
    """
    Function that produces a dataframe with frequency of features by class and returns lists with features names by class.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :param drop_vars: list of support columns.
    :type drop_vars: list.

    :param drop_excessive_miss: flag indicating whether columns with excessive missings should be dropped out.
    :type drop_excessive_miss: boolean.

    :param drop_no_var: flag indicating whether columns with no variance should be dropped out.
    :type drop_no_var: boolean.

    :param validation_data: additional data.
    :type validation_data: dataframe.

    :param test_data: additional data.
    :type test_data: dataframe.

    :return: dataframe and lists with features by class.
    :rtype: dictionary.
    """
    feats_assess_dict = {}
    
    print(f'Initial number of features: {dataframe.drop(drop_vars, axis=1).shape[1]}.')

    if drop_excessive_miss:
        # Dropping features with more than 95% of missings in the training data:
        excessive_miss_train = [c for c in dataframe.drop(drop_vars, axis=1) if
                                sum(dataframe[c].isnull())/len(dataframe) > 0.95]

        if len(excessive_miss_train) > 0:
            dataframe.drop(excessive_miss_train, axis=1, inplace=True)

            if validation_data is not None:
                validation_data.drop(excessive_miss_train, axis=1, inplace=True)
                
            if test_data is not None:
                test_data.drop(excessive_miss_train, axis=1, inplace=True)

        print(f'{len(excessive_miss_train)} features were dropped for excessive number of missings!')

    # Data type of each feature:
    feats_class = pd.DataFrame(data = {
        'feature': dataframe.drop(drop_vars, axis=1).dtypes.index,
        'class': dataframe.drop(drop_vars, axis=1).dtypes.values
    })

    # Categorical features
    cat_vars = list(feats_class[feats_class['class']=='object'].feature)

    if drop_no_var:
        # Dropping features with no variance in the training data:
        no_variance = [c for c in dataframe.drop(drop_vars, axis=1).drop(cat_vars,
                                                                         axis=1) if dataframe[c].var()==0]

        if len(no_variance) > 0:
            dataframe.drop(no_variance, axis=1, inplace=True)
            if validation_data is not None:
                validation_data.drop(no_variance, axis=1, inplace=True)
                
            if test_data is not None:
                test_data.drop(no_variance, axis=1, inplace=True)

        print(f'{len(no_variance)} features were dropped for having no variance!')

    # Numerical features:
    cont_vars = [c for c in  list(dataframe.drop(drop_vars, axis=1).columns) if is_velocity(c)]

    # Binary features:
    binary_vars = [c for c in list(dataframe.drop([c for c in dataframe.columns if (c in drop_vars) |
                                                        (c in cat_vars) | (c in cont_vars)],
                                                        axis=1).columns) if
                   set(dataframe[~dataframe[c].isnull()][c].unique()) == set([0,1])]

    # Updating the list of numerical features:
    for c in list(dataframe.drop(drop_vars, axis=1).columns):
        if (c not in cat_vars) & (c not in cont_vars) & (c not in binary_vars):
            cont_vars.append(c)

    print(f'{dataframe.drop(drop_vars, axis=1).shape[1]} remaining features.')
    print('\n')


    # Dataframe presenting the frequency of features by class:
    feats_assess = pd.DataFrame(data={
        'class': ['cat_vars', 'binary_vars', 'cont_vars', 'drop_vars'],
        'frequency': [len(cat_vars), len(binary_vars), len(cont_vars), len(drop_vars)]
    })
    feats_assess.sort_values('frequency', ascending=False, inplace=True)
    
    # Dictionary with outputs from the function:
    feats_assess_dict['feats_assess'] = feats_assess
    feats_assess_dict['cat_vars'] = cat_vars
    feats_assess_dict['binary_vars'] = binary_vars
    feats_assess_dict['cont_vars'] = cont_vars
    
    if drop_excessive_miss:
        feats_assess_dict['excessive_miss_train'] = excessive_miss_train

    if drop_no_var:
        feats_assess_dict['no_variance'] = no_variance
    
    return feats_assess_dict

####################################################################################################################################
# Function that produces an assessment of the occurrence of missing values:

def assessing_missings(dataframe):
    """
    Function that produces an assessment of the occurrence of missing values.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :return: dataframe with frequency and share of missings by feature.
    :rtype: dataframe.
    """
    # Dataframe with the number of missings by feature:
    missings_dict = dataframe.isnull().sum().sort_values(ascending=False).to_dict()

    missings_df = pd.DataFrame(data={
        'feature': list(missings_dict.keys()),
        'missings': list(missings_dict.values()),
        'share': [m/len(dataframe) for m in list(missings_dict.values())]
    })

    print('\033[1mNumber of features with missings:\033[0m {}'.format(sum(missings_df.missings > 0)) +
          ' out of {} features'.format(len(missings_df)) +
          ' ({}%).'.format(round((sum(missings_df.missings > 0)/len(missings_df))*100, 2)))
    print('\033[1mAverage number of missings:\033[0m {}'.format(int(missings_df.missings.mean())) +
          ' out of {} observations'.format(len(dataframe)) +
          ' ({}%).'.format(round((int(missings_df.missings.mean())/len(dataframe))*100,2)))
    
    return missings_df

####################################################################################################################################
# Function that forces consistency between reference (training) and additional (validation, test) data:

def data_consistency(dataframe, *args, **kwargs):
    """
    Function that forces consistency between reference (training) and additional (validation, test) data:

    The keyword arguments are expected to be dataframes whose argument names indicate the nature of the passed data. For instance,
    'test_data=df_test' would be a dataframe with test instances.

    :param dataframe: reference data.
    :type dataframe: dataframe.

    :return: dataframes with consistent data.
    :rtype: dictionary.
    """
    consistent_data = {}
    
    for d in kwargs.keys():
        consistent_data[d] = kwargs[d].copy()
        
        # Columns absent in reference data:
        absent_train = [c for c in kwargs[d].columns if c not in dataframe.columns]
        
        # Columns absent in additional data:
        absent_test = [c for c in dataframe.columns if c not in kwargs[d].columns]
        
        # Creating absent columns:
        for c in absent_test:
            consistent_data[d][c] = 0
    
        # Preserving consistency between reference and additional data:
        consistent_data[d] = consistent_data[d][dataframe.columns]
        
        # Checking consistency:
        if sum([1 for r, a in zip(dataframe.columns, consistent_data[d].columns) if r != a]):
            print('Problem - Reference and additional datasets are inconsistent!')
        else:
            print(f'Training and {d.replace("_", " ")} are consistent with each other.')
    
    return consistent_data

####################################################################################################################################
# Function that returns the amount of time for running a block of code:

def running_time(start_time, end_time, print_time=True):
    """
    Function that returns the amount of time for running a block of code.
    
    :param start_time: time point when the code was initialized.
    :type start_time: datetime object obtained by executing "datetime.now()".

    :param end_time: time point when the code stopped its execution.
    :type end_time: datetime object obtained by executing "datetime.now()".

    :param print_unit: unit of time for presenting the runnning time.
    :type print_unit: string.
    
    :return: prints start, end time and running times, besides of returning running time in seconds.
    :rtype: integer.
    """
    if print_time:
        print('------------------------------------')
        print('\033[1mRunning time:\033[0m ' + str(round(((end_time - start_time).total_seconds())/60, 2)) +
              ' minutes.')
        print('Start time: ' + start_time.strftime('%Y-%m-%d') + ', ' + start_time.strftime('%H:%M:%S'))
        print('End time: ' + end_time.strftime('%Y-%m-%d') + ', ' + end_time.strftime('%H:%M:%S'))
        print('------------------------------------')
    
    return (end_time - start_time).total_seconds()

####################################################################################################################################
# Function that calculates frequencies of elements in a list:

def frequency_list(_list):
    """
    Function that calculates frequencies of elements in a list:
    
    :param _list: .
    :type _list: list.
    
    :return: dictionary whose keys are the elements in a list and values are their frequencies.
    :rtype: dictionary.
    """
    _set = set(_list)
    freq_dict = {}

    # Loop over unique elements:
    for f in _set:
        freq_dict[f] = 0

        # Counting frequency:
        for i in _list:
            if i == f:
                freq_dict[f] += 1
    
    return freq_dict

####################################################################################################################################
# Function that assess the number of missings in a dataframe:

def missings_detection(dataframe, name='df', var=None):
    """"
    Function that assess the number of missings in a dataframe

    :param dataframe: dataframe for which missings should be detected.
    :type dataframe: dataframe.
    
    :param name: name of dataframe.
    :type name: string.
    
    :param var: name of variable whose missings should be detected (optional).
    :type var: string.

    :return: prints the number of missings when there is a positive amount of them.
    """

    if var:
        num_miss = dataframe[var].isnull().sum()
        if num_miss > 0:
            print(f'Problem - There are {num_miss} missings for "{var}" in dataframe {name}.')

    else:
        num_miss = dataframe.isnull().sum().sum()
        if num_miss > 0:
            print(f'Problem - Number of overall missings detected in dataframe {name}: {num_miss}.')

####################################################################################################################################
# Function that apply the statistical test for difference between proportions:

def applying_prop_test(dataframe, reference_var, outcome_var='y', alternative='two-sided'):
    """
    Function that apply the statistical test for difference between proportions (means of a binary variable) for
    two different samples.
    
    This function makes use of "proportions_ztest" function from "statsmodels" library. In the way it is
    implemented here, the following hypothesis are tested against each other:
        H0: P(outcome_var = 1|reference_var = 1) = P(outcome_var = 1|reference_var = 0)
        H1: P(outcome_var = 1|reference_var = 1) != P(outcome_var = 1|reference_var = 0)
    
    Where the variable "reference_var" is responsible for splitting a dataset into two different samples, while
    "outcome_var" is the binary variable of interest.
    
    :param dataframe: dataframe with samples for implementing the test.
    :type dataframe: dataframe.
    
    :param reference_var: binary variable that will split samples accross two subsets.
    :type reference_var: string.
    
    :param outcome_var: binary variable whose difference in proportion should be assessed.
    :type outcome_var: string.
    
    :param alternative: indicate whether the test is two-sided (p0 != p1) or one-sided ("smaller" for p0 < p1,
    "larger" for p1 > p0).
    :type alternative: string.
    
    :return: test statistic, p-value of the test, hypotheses being tested, relevant frequencies.
    :rtype: dictionary.
    """
    
    oper = '<' if alternative=='smaller' else ('>' if alternative=='larger' else '!=')
    
    d0 = len(dataframe[reference_var]) - dataframe[reference_var].sum()
    d1 = dataframe[reference_var].sum()

    d0_y1 = len(dataframe[(dataframe[reference_var]==0) & (dataframe[outcome_var]==1)][reference_var])
    d1_y1 = len(dataframe[(dataframe[reference_var]==1) & (dataframe[outcome_var]==1)][reference_var])

    count = np.array([d0_y1, d1_y1])
    nobs = np.array([d0, d1])
    stat, pval = proportions_ztest(count, nobs, alternative=alternative)
    
    return {'test_stat': stat, 'p_value': pval,
            'hypotheses': f'H0: P({outcome_var}=1|{reference_var}=0) = P({outcome_var}=1|{reference_var}=1)\n'\
                          f'H1: P({outcome_var}=1|{reference_var}=0) {oper} P({outcome_var}=1|{reference_var}=1)',
            'frequencies': {f'freq({reference_var}=0)': d0,
                            f'freq({reference_var}=1)': d1,
                            f'freq({reference_var}=0&{outcome_var}=1)': d0_y1,
                            f'freq({reference_var}=1&{outcome_var}=1)': d1_y1}}

####################################################################################################################################
# Function that calculates cross-entropy given true labels and predictions:

def cross_entropy_loss(y_true, p):
    prediction = np.clip(p, 1e-14, 1. - 1e-14)
    return -np.sum(y_true*np.log(prediction) + (1-y_true)*np.log(1-prediction))/len(y_true)
