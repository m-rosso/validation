# Validation procedures

This projects conducted to the development of the following classes that have the goal of contributing with validation procedures during the implementation of data modeling in supervised learning tasks:
1. **KfoldsCV**: executes K-folds cross-validation combined with grid or random searches for defining hyper-parameters of several distinct machine learning algorithms.

2. **KfoldsCV_fit**: extends the first to include the estimation of a model using the entire training data and the best choices of hyper-parameters, in addition to providing estimates of performance metrics evaluated on test data. This class inherits from "KfoldsCV", mainly because both initializations are the same. The main differences between these classes are that the first does not allow fitting a final model, and that no test data is provided to it.

3. **BootstrapEstimation**: allows the implementation of bootstrap estimations. The core of this procedure is to run a large collection of estimations, where each one of them is based on a different sample drawn from the training data with the same length of it and considering replacement. An alternative to that, also available using this class, is to produce samples without replacement (averaging), so estimations do not follow the bootstrap principle, but still permit the calculation of statistics for performance metrics.

The first two classes are specially relevant for empirical contexts where available data has no sufficient length to provide appropriate batches of training, validation, and test data. Consequently, the fine tuning of hyper-parameters should be implemented based solely on the training data. The last class makes use of "KfoldsCV", being particularly useful when data modeling is subject to high variability, such as high-dimensional problems or the estimation of models based on decision trees (GBM, random forest, and so on). Since both KfoldsCV_fit and bootstrap_estimation inherit from KfoldsCV, this class has the most relevant documentation, as it introduces several distinct initialization parameters that apply for all of these classes.

K-folds cross-validation allows appropriate predictions for training data instances, yet its main application consists of exploring different combinations of values for relevant hyper-parameters. There are two main alternatvies for performing fine tuning under this validation procedure: grid search and random search. Both are available here by making use of "grid_param" and "random_search" initialization parameters.

KfoldsCV and all classes that inherit from it allow a large amount of control over **hyper-parameters**. There are three initialization parameters relevant for this: "grid_param", a dictionary with hyper-parameters as keys and corresponding values to be explored; "default_param", a dictionary with values of hyper-parameters to be used when some possibilities fail to return valid results during grid/random search; and "fixed_params", hyper-parameters that should not be explored during optimization, but instead of using libraries default values, it is possible to choose them appropriately as a function of the application.

Setting "random_search" to False and declaring "grid_param", one can execute **grid search**, which is more likely to provide fine results when previous studies have been done so only most promising alternatives remain left. Setting "random_search" to True and declaring "grid_param" as a dictionary whose keys are the most relevant hyper-parameters of a ML method and whose respective values are lists with integer values or statistical distributions from scipy.stats (such as "uniform", "norm", "randint"), then **random search** can be implemented. This alternative is specially suited for ML methods that have multiple hyper-parameters. Finally, when random_search is set to True, "n_samples" should be declared for the number of randomly picked combinations of hyper-parameters.

"task" and "method" are the two fundamental parameters for initializing those classes. Supervised learning tasks available using these classes are binary classification and regression. Some slight modifications are required for implementing multiclass classification (which should be done soon). For these two problems, the following ML methods are supported:
1. Logistic regression from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (method='logistic_regression').
    * Main hyper-parameters for tuning: regularization parameter ('C').
2. Linear regression (Lasso) from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (method='lasso').
    * Main hyper-parameters for tuning: regularization parameter ('C').
3. GBM from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) (method='gbm').
    * Hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('learning_rate'), number of estimators ('n_estimators').
4. GBM from [LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html) (method='light_gbm').
    * Main hyper-parameters for tuning: subsample ('bagging_fraction'), maximum depth ('max_depth'), learning rate ('learning_rate), number of estimators ('num_iterations').
    * By declaring 'metric' and 'early_stopping_rounds' into the parameters dictionary, it is possible to implement both "KfoldsCV" and "Kfolds_fit" with early stopping. For "KfoldsCV", at each k-folds estimation early stopping will take place, while for "Kfolds_fit" estimation will stop after a stopping rule is triggered both during each of k-folds estimation and during the final fitting using the entire training data.
5. GBM from [XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html#xgboost-parameters) (method='xgboost').
    * Main hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('eta'), number of estimators ('num_boost_round').
    * By declaring 'eval_metric' and 'early_stopping_rounds' into the parameters dictionary, also for XGBoost early stopping is available for both "KfoldsCV" and "Kfolds_fit".
6. Random forest from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) (method='random_forest').
    * Main hyper-parameters for tuning: number of estimators ('n_estimators'), maximum number of features ('max_features') and minimum number of samplesfor split ('min_samples_split').
7. SVM from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (method='svm').
    * Main hyper-parameters for tuning: regularization parameter ('C') kernel ('kernel'), polynomial degree ('degree'), gamma ('gamma').

**Performance metrics** allowed for binary classification are ROC-AUC, average precision score (as proxy for precision-recall AUC), and Brier score. For regression, RMSE, MAE, R2 and log-MSE are the metrics available. These metrics are used for optimizing hyper-parameters and also for reference in terms of evaluating a trained model.

Declaring "parallelize" equal to True when initializing the K-folds CV object is expected to improve running time, since training-validation estimation for all K folds of data is then implemented in **parallel**.

KfoldsCV and classes that inherit from it cover **features selection**. First, one may choose among three different classes of features selection methods: analytical methods (variance and correlation thresholding), supervised learning selection (picking features according to feature importances as provided by supervised learning methods) and exaustive methods (RFE, RFECV, sequential selection, random selection). Second, when using "KfoldsCV_fit" class, one may choose between applying features selection at each iteration of K-folds cross-validation or only when final model is trained. These features selection tools derived from "FeaturesSelection" class, which in its turn follows from some sklearn classes and some independently developed classes and functions.

The development of these classes had the **main objectives** of improving statistical learning theory learnt by me through the past years, and (believe it or not) having some fun applying programming and statistical concepts to develop something from scratch. Notwithstanding, some interesting features have emerged and are available by making use of these classes:

1. *Simplified usage:* when it comes to applying K-folds cross-validation, the interface here is maybe more intuitive, since it requires only the creation of an object from "KfoldsCV" or from "KfoldsCV_fit". All relevant parameters are then initialized, and only a method for running/fitting should be executed having as arguments only the data. Finally, all relevant outcomes from executing these validation methods are kept together with the object as attributes.

2. *Unified framework:* all classes are centered around the same K-folds CV principle, even allowing the construction of a robust bootstrap estimation class.

3. *Additional features:* more information is provided by the end of estimations, such as cross-validation predictions for training data, performance metrics, progress bar and elapsed time.

4. *Pre-selection of input variables:* pre-selecting features inside K-folds CV allows to run random/grid search without the fear of incurring in biases from selecting features in a first place and then proceeding to other supervised learning tasks. When it comes to pre-selection of features, all these classes are very flexibly, both in methods available and in terms of it usage.

5. *More flexibility:* by changing components of method "\__create_model", models from any library can be applied, not only those provided by sklearn, all in the same framework. Currently, LightGBM and XGBoost are available, but also neural networks from Keras should probably be inserted soon.
