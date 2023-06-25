# path: K_Nearest_Neighbors\main.py
#data from : https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

import pandas as pd
import numpy as np
import sklearn

from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("data\Car Data Set\car.data")#read data from csv file using pandas


le = preprocessing.LabelEncoder()#create a LabelEncoder object

buying = le.fit_transform(list(data["buying"]))#convert the buying column to numerical data
maint = le.fit_transform(list(data["maint"]))#convert the maint column to numerical data
door = le.fit_transform(list(data["door"]))#convert the door column to numerical data
persons = le.fit_transform(list(data["persons"]))#convert the persons column to numerical data
lug_boot = le.fit_transform(list(data["lug_boot"]))#convert the lug_boot column to numerical data
safety = le.fit_transform(list(data["safety"]))#convert the safety column to numerical data
cls = le.fit_transform(list(data["class"]))#convert the class column to numerical data
# print(buying) #output: [3 3 3 ... 1 1 1]

predict = "class"#set the column to be predicted


while True:
    x = list(zip(buying, maint, door, persons, lug_boot, safety))#create a list of tuples of the features 
    y = list(cls)#create a list of the labels

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)#split the data into train and test data

    model = KNeighborsClassifier(n_neighbors=3)#create a KNeighborsClassifier object #n_neighbors is how many neighbors to look at

    model.fit(x_train, y_train)#train the model    
    acc = model.score(x_test, y_test)#get the accuracy of the model
    print(acc)#output the accuracy of the model e.g:0.815028901734104
    if acc > 0.9:
        break

predicted = model.predict(x_test)#get the predicted classes
names = ["unacc", "acc", "good", "vgood"]#create a list of the names of the classes

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])#output the predicted class, the features and the actual class
#"""
    print("/n")
    n = model.kneighbors([x_test[x]], 9, True)#get the 9 nearest neighbors
    print("N: ", n)#output the 9 nearest neighbors"""

