# path: linear_regression\main.py
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle

from matplotlib import style
from sklearn import linear_model

### Linear Regression ###
def linear_regression():
    data = pd.read_csv("data\student\student-mat.csv", sep=";")#read data from csv file using pandas, the data some how is seperated by ;

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
        """
        pickle_in = open("studentmodel.pickle", "rb")#load the model
        linear = pickle.load(pickle_in)

        print("Coefficient: \n", linear.coef_)#get the coefficient of the model
        print("Intercept: \n", linear.intercept_)#get the intercept of the model

        predictions = linear.predict(x_test)#get the predictions of the model

        for x in range(len(predictions)):#loop through the predictions
            print(predictions[x], x_test[x], y_test[x])#print the prediction, the test data and the actual data
        """

    print("Best Accuracy: ", best)

def run():
    data = pd.read_csv("data\student\student-mat.csv", sep=";")#read data from csv file using pandas, the data some how is seperated by ;

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

if __name__ == "__main__":
    #linear_regression()
    run()