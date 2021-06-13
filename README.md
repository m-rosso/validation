# Validation procedures

This projects conducted to the development of the following Python classes that have the goal of contributing with validation procedures during the implementation of data modeling in supervised learning tasks:
1. **KfoldsCV**: executes K-folds cross-validation combined with grid or random searches for defining hyper-parameters of several distinct machine learning algorithms.
2. **KfoldsCV_fit**: extends the first to include the estimation of a model using the entire training data and the best choices of hyper-parameters, in addition to providing estimates of performance metrics evaluated on test data. This class inherits from "KfoldsCV", mainly because both initializations are the same. The main differences between these classes are that the first does not allow fitting a final model, and that no test data is provided to it.
3. **bootstrap_estimation**: allows the implementation of bootstrap estimations. The core of this procedure is to run a large collection of estimations, where each one of them is based on a different sample drawn from the training data with the same length of it and considering replacement. An alternative to that, also available using this class, is to produce samples without replacement (averaging), so estimations do not follow the bootstrap principle, but still permit the calculation of statistics for performance metrics.

The first two classes are specially relevant for empirical contexts where available data has no sufficient length to provide appropriate batches of training, validation, and test data. Consequently, the fine tuning of hyper-parameters should be implemented based solely on the training data. The last class makes use of "KfoldsCV", being particularly useful when data modeling is subject to high variability, such as high-dimensional problems or the estimation of models based on decision trees (GBM, random forest, and so on).

Both KfoldsCV_fit and bootstrap_estimation inherit from KfoldsCV, so this class has the most relevant documentation, since it introduces several distinct initialization parameters that apply for all of these classes. One relevant attribute of KfoldsCV, for instance, is *parallelize*, which, when declared equal to True, is expected to improve running time, since training-validation estimation for all K folds of data is then implemented in parallel.

Supervised learning tasks available using these classes are binary classification and regression. Some slight modifications are
required for implementing multiclass classification (which should be done soon). For these two problems, the following **ML methods**
are supported:
1. Logistic regression (from sklearn).
    * Hyper-parameters for tuning: regularization parameter ('C').
2. Linear regression (Lasso) (from sklearn).
    * Hyper-parameters for tuning: regularization parameter ('C').
3. GBM (sklearn).
    * Hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('learning_rate'),
    number of estimators ('n_estimators').
4. GBM (LightGBM).
    * Hyper-parameters for tuning: subsample ('bagging_fraction'), maximum depth ('max_depth'), learning rate ('learning_rate'),
    number of estimators ('num_iterations').
5. GBM (XGBoost).
    * Hyper-parameters for tuning: subsample ('subsample'), maximum depth ('max_depth'), learning rate ('eta'), number of estimators ('num_boost_round').
6. Random forest (from sklearn).
    * Hyper-parameters for tuning: number of estimators ('n_estimators'), maximum number of features ('max_features') and minimum
    number of samplesfor split ('min_samples_split').
7. SVM (from sklearn).
    * Hyper-parameters for tuning: regularization parameter ('C') kernel ('kernel'), polynomial degree ('degree'), gamma ('gamma').

**Performance metrics** allowed for binary classification are ROC-AUC, average precision score (as proxy for precision-recall AUC), and Brier score. For regression, RMSE is the metric available by now.
<br>
<br>
The development of these classes had the **main objectives** of improving statistical learning theory learnt by me through the past years, and (believe it or not) having some fun applying programming and statistical concepts to develop something from scratch. Notwithstanding, some interesting **features** have emerged and are available by making use of these classes:
1. *Simplified usage:* when it comes to applying K-folds cross-validation, the interface here is maybe more intuitive, since it requires only the creation of an object from "KfoldsCV" or from "KfoldsCV_fit". All relevant parameters are then initialized, and only a method for running/fitting should be executed having as arguments only the data. Finally, all relevant outcomes from executing these validation methods are kept together with the object as attributes.
2. *Unified framework:* all classes are centered around the same K-folds CV principle, even allowing the construction of a robust bootstrap estimation class.
3. *Additional features:* more information is provided by the end of estimations, such as cross-validation predictions for training data, performance metrics, progress bar and elapsed time.
4. *Pre-selection of input variables:* the use of statistical learning methods (here, logistic regression or Lasso) for features selection is available. Even that all ML methods have some kind of regularization, this should help filtering out non-relevant features for some specific empirical context. If this kind of features selection is expected to happen in final model estimation, it should also be internatilized during definition of hyper-parameters.
5. *More flexibility:* by changing components of method "\__create_model", models from any library can be applied, not only those provided by sklearn, all in the same framework. Currently, Light GBM is available, but also XGBoost and neural networks from Keras should probably be inserted soon.
