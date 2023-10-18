---
layout: archive-dates
permalink: /mldl/
title: Portfolio of Machine Learning(ML), Deep Learning(DL), Time Series(TS) and NLP Projects
---

In the following links, you can check out some interesting Machine learning, Deep Learning and Time series forecasting models I had worked with!

## ML & DL Models
-------------

### Master of Science Thesis 

This study concentrates on predicting regional traffic density through the analysis of sensor data from Madrid's URB and M30 roads at 15-minute intervals in 21 districts of Madrid city between January 2022 and June 2022. The pre-processed data was inputted into deep learning models as numerical time-distributed series and generated hourly traffic intensity map images. Using CNN and LSTM structures within an encoder-decoder model, the study aimed for sequence-to-sequence prediction, with 24-time step looking back and 4-time step forecasting range. The models outperformed the baseline, with spatial correlation enhancing accuracy, and a Bi-LSTM encoder-decoder model further optimized predictions.

- [Deep Learning Models to Predict the Traffic Intensity in Madrid city](/Notebooks/MasterThesis.pdf)

### Bachelor of Science Thesis 

This thesis is about the asset amount prediction of the bank customer with the available data (Bank Data Warehouse, KKB, Neighborhood data). In that way, it is aimed to make the right campaign for the right customer. In order to establish this predictive model, the main goal is to use machine learning algorithms and to make the model in a software program that supports these algorithms.

In this thesis, four machine learning algorithms that were examined in the literature review header used to predict the asset of bank’s customers according to many conditions created in Oracle SQL software. Model performance was measured with MAE and MAPE KPI forecast error types. After examination of MAE and MAPE values for test samples and also model performance graphs, the XGBoost algorithm was chosen as an optimal model for this case. Test MAE and MAPE values were found 98.642, 17% respectively.

- [Asset Estimation of the Bank Customers via Machine Learning Approach](/Notebooks/bachelor_thesis.pdf)

### LeNet-5 CNN Network Implementation

The architecture of the LeNet-5 Network can be seen on the following visual:

<img src="/images/lenet5.jpeg?raw=true"/>

As seen on the visual, the LeNet-5 architecture consists of *two convolutional and average pooling layers*, followed by a *flattening convolutional layer*. After these layers, it has *two fully-connected layers* and finally a *softmax* classifier. In the following link, it can be seen the implementation in PyTorch.

- [LeNet 5 CNN Network](/Notebooks/LeNet5_CNN.html)


### Prediction of House Prices [(Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Many different methods (KNN Regressor, Decision Trees, Extra trees, Random forest, HistGradient Regressor and XGBoost) were implemented to predict the house prices. The best optimal model was chosen by doing cross validation with 3 folds. To improve the performance of each model, different techniques were implemented for hyper-parameter tuning such as Grid search, Random search, Bayes search, Optuna, Halving Grid Search and Halving Random Search.

- [Kaggle ML Prediction Competition](/Notebooks/Kaggle_Comp.html)

<img src="/images/ml1.PNG?raw=true"/>

### Gaussian Processes

To see the implementation of Gaussian processes with GPy, click on the following link:

- [Regression with Gaussian Processes](/Notebooks/GPs.md)


### Comparison of different kernel methods for multiclass classification

In this study, five different schema were implemented for face recognition classification. 

- [Kernel Methods](/Notebooks/kernels.html)


## Time Series Forecasting Statistical & ML Models
-------------

Here, I have placed some projects about time series forecasting with statistical and Machine learning models.

### Prediction of hourly traffic intensity in Madrid city

In this project, the traffic insensity level of the most crowded district of Madrid city was estimated in every hour by implementing both statistical and Machine Learning models. The data used in this project was obtained from the sensors placed in Madrid city.

- [Hourly Traffic Intensity Forecasting in Madrid](Notebooks/hourly_traffic_pred.html)

### Time Series Forecasting of the annual exports of Turkey

In this study, the annual exports of Turkey has been analyzed and exports has been forecasted by statistical tools such as ETS.

- [Annual exports of Turkey](Notebooks/turkey_annual_exports.html)

## Natural Language Processing Topic Model Prediction

In the following link, you can check out the project related with topic models prediction in python!

- [Topic models Prediction](/Notebooks/Topic_models.html)
