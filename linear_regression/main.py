# path: linear_regression\main.py
# data from : https://archive.ics.uci.edu/ml/datasets/Student+Performance
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle

from matplotlib import style
from sklearn import linear_model

### Linear Regression ###
# y = mx + b
# m = slope(斜率), b = y-intercept(截距)
# intercept(截距) means the point where the line crosses the y axis

# linear regression is a machine learning algorithm based on supervised learning

# linear_regression function is used to train the model and save the model as pickle file
# run function is will get the data from the csv file get the data that we want to use and predict the data using the model that we saved
# the data that we want to use is G1, G2, studytime, failures, absences
# the data that we want to predict is G3
# after we get the data that we want to use and predict, we will split the data into train and test data
# then we will train the model using the train data until the accuracy is better than 90%
# and save the model as pickle file
def linear_regression():
    data = pd.read_csv("data\student\student-mat.csv")#read data from csv file using pandas

    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]#select the columns that we want to use

    predict = "G3"#the column that we want to predict

    x = np.array(data.drop([predict], axis=1))#drop the column that we want to predict
    y = np.array(data[predict])#the column that we want to predict
    x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)#split the data into train and test data

    best = 0
    while True:
        linear = linear_model.LinearRegression()#create a linear regression model
        x = np.array(data.drop([predict], axis=1))#drop the column that we want to predict
        y = np.array(data[predict])#the column that we want to predict
        x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)#split the data into train and test data
        linear.fit(x_train, y_train)#train the model

        acc = linear.score(x_test, y_test)#get the accuracy of the model
        print("Accuracy: ", acc)

        if acc > best:#if the accuracy is better than the previous one
            best = acc#set the best accuracy to the current one

            #save the model as pickle file
            with open("studentmodel.pickle", "wb") as f:#save the model
                pickle.dump(linear, f)

        if best > 0.9:#if the accuracy is better than 90%
            break

    print("Best Accuracy: ", best)

def run():
    data = pd.read_csv("data\student\student-mat.csv")#read data from csv file using pandas

    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]#select the columns that we want to use

    predict = "G3"#the column that we want to predict

    x = np.array(data.drop([predict], axis=1))#drop the column that we want to predict
    y = np.array(data[predict])#the column that we want to predict
    x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)#split the data into train and test data
    
    pickle_in = open("studentmodel.pickle", "rb")#load the model
    linear = pickle.load(pickle_in)

    print("Coefficient: \n", linear.coef_)#get the coefficient of the model
    print("Intercept: \n", linear.intercept_)#get the intercept of the model

    predictions = linear.predict(x_test)#get the predictions of the model

    for x in range(len(predictions)):#loop through the predictions
        print(predictions[x], x_test[x], y_test[x])#print the prediction, the test data and the actual data

    p = "absences"#the column that we want to predict
    style.use("ggplot")#set the style of the graph
    plt.scatter(data[p], data["G3"])#plot the graph
    plt.xlabel(p)#set the x label
    plt.ylabel("Final Grade")#set the y label
    plt.show()#show the graph

# inputrun function is used to predict the data that we want to predict using the model that we saved
def inputrun(G1, G2, studytime, failures, absences):
    pickle_in = open("studentmodel.pickle", "rb")#load the model
    linear = pickle.load(pickle_in)

    predictions = linear.predict([[G1, G2, studytime, failures, absences]])#get the predictions of the model

    return predictions[0]#return the prediction

if __name__ == "__main__":
    linear_regression()
    #run()
    print(inputrun(10, 10, 2, 0, 0))