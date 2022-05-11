# Flexible Naive Bayes
A comparative study between Naive Bayes Classifier and Flexible Bayes Classifier. 4 bandwidth selection methods are used: Scott's Factor, Silverman's rule of thumb, Least Square Cross Validation and Maximum Likelihood Cross Likelihood.

## Simulation
Data generated from both Gaussian and non Gaussian Distribution and compared the predictive accuracies between Naive Bayes and Flexible Bayes Classifier.

### 1. Gaussian Distribution
Data are generated from a multivariate gaussian distribution where the location parameter is (1, 2, 3, 4, 5)' and scale parameter is 1.

* Number of classes is 5
* Class sizes are respectively 100, 110, 95, 101, 99.

|Classifier|Train Accuracy|Test Accuracy|
|:-:|:-:|:-:|
|Naive Bayes |0.8237|0.8823|
|Flexible Bayes (scott)|0.9801|0.8431|
|Flexible Bayes (Silverman)|0.9845|0.8627|
|Flexible Bayes (CV-LS)|0.9713|0.7647|
|Flexible Bayes (CV-ML)|0.9581|0.8039|

### 2. Non-Gaussian Distribution
Data are generated from a multivariate laplace distribution with the location parameter is (1, 2, 3, 4, 5)' and scale parameter 1.

* Number of classes is 5
* Class sizes are respectively 100, 110, 95, 101, 99.

|Classifier|Train Accuracy|Test Accuracy|
|:-:|:-:|:-:|
|Naive Bayes |0.7224|0.6274|
|Flexible Bayes (scott)|0.9647|0.6274|
|Flexible Bayes (Silverman)|0.9735|0.6470|
|Flexible Bayes (CV-LS)|0.9669|0.5490|
|Flexible Bayes (CV-ML)|0.9339|0.6862|

## Real Life Data Analysis
For Real life data, the iris dataset, wisconsin breast cancer dataset and wine datasets are used.

### 1. Wisconsin Breast Cancer Dataset
* Number of classes is 2
* Total number of observations is 569

|Classifier|Train Accuracy|Test Accuracy|
|:-:|:-:|:-:|
|Naive Bayes |0.9342|0.9301|
|Flexible Bayes (scott)|0.9953|0.9370|
|Flexible Bayes (Silverman)|0.9976|0.9510|
|Flexible Bayes (CV-LS)|1.0000|0.4196|
|Flexible Bayes (CV-ML)|0.9765|0.9440|

### 2. Iris Dataset
* Number of classes is 3
* Number of observations is 150

|Classifier|Train Accuracy|Test Accuracy|
|:-:|:-:|:-:|
|Naive Bayes |0.9464|1.0000|
|Flexible Bayes (scott)|1.000|1.000|
|Flexible Bayes (Silverman)|1.000|1.000|
|Flexible Bayes (CV-LS)|1.0000|0.7894|
|Flexible Bayes (CV-ML)|0.9910|0.9736|

### 3. Wine Dataset
* Number of classes is 3
* Number of observations is 178

|Classifier|Train Accuracy|Test Accuracy|
|:-:|:-:|:-:|
|Naive Bayes |0.9924|0.9333|
|Flexible Bayes (scott)|1.0000|1.0000|
|Flexible Bayes (Silverman)|1.0000|1.0000|
|Flexible Bayes (CV-LS)|0.9849|0.6444|
|Flexible Bayes (CV-ML)|1.0000|0.9778|