# path: svm/main.py
# svm means support vector machine

# support vector machine is a supervised machine learning algorithm 
#which can be used for both classification or regression challenges.
# However,  it is mostly used in classification problems.
# In this algorithm, we plot each data item as a point in n-dimensional space

import sklearn
from sklearn import svm
from sklearn import datasets

# load data
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train, y_train)
class = ['malignant', 'benign']# change 0,1 to malignant, benign
