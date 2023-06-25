#path: data\student\change_the_csv.py
#some hot csv files split by ; instead of ,
#so this code is used to change the ; to ,
#do this coz i need to use the vscode Excel Viewer to view the csv file >_<

import pandas as pd


data = pd.read_csv("data\student\student-por.csv", sep=";")#read data from csv file using pandas, the data some how is seperated by ;

data.to_csv("data\student\student-por.csv", sep=",", index=False)#save the data as csv file, the data is seperated by , and the index is not included in the csv file

print("Done")#print Done when the program is done

