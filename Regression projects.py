# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:56:47 2022

@author: mmoein2
"""
#Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #split data in training and testing set
from sklearn.model_selection import cross_val_score #K-fold cross validation
from sklearn import linear_model #Importing both linear regression and logistic regression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

#Ask for file name
file_name = 'TSS'
file_header = input("File has labels and header (Y):")


#Create a pandas dataframe from the csv file.      
if file_header == 'Y' or file_header == 'y':
    data = pd.read_csv(file_name + '.csv', header=0, index_col=0) #Remove index_col = 0 if rows do not have headers
else:
    data = pd.read_csv(file_name + '.csv', header=None)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns.values)))
print('')


#Defining Xs, and Y
X = data.drop(columns=['TSS'])
Y = data[["TSS"]]

#############################################
#Scaling or not the data
#from sklearn import preprocessing to scale the values
#############################################


#Using Built in train test split function in sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 , stratify=Y.TSS)


#Perform linear regression
while True:    
    lin_reg = input("Perform a linear regression (Y):")
    if lin_reg == 'Y' or lin_reg == 'y':
        print('') #Add onle line for space
        print(data.columns.values)
        #Set the x and y variables.
        x_var = input("X variable:")
        y_var = input("Y variable:")
        print('')
        #sklearn prefers 2d arrays, so we simply change the format of the data using the "reshape" function.
        x = data[x_var].values.reshape(-1, 1)
        y = data[y_var].values.reshape(-1, 1)
        
        lr = linear_model.LinearRegression() #Call linear regression model
        lr.fit(x, y) #Fit regression
        print("{0} * x + {1}".format(lr.coef_[0][0].round(2), lr.intercept_[0].round(2))) #Show equations calculated
        
        y_pred = lr.predict(x) #Calculate points predicted to measure accuracy
        R2 = metrics.r2_score(y, y_pred) #Calculate R2 score
        print("MSE: {0} and R2: {1}".format(np.mean((y_pred - y)**2).round(2), R2.round(2))) #round(xxx,2)
        
    else:
        break

#Perform multivariate linear regression
multi_reg = input("Perform a multi-variate linear regression (Y):")
if multi_reg == 'Y' or multi_reg == 'y':
    Y_mreg = Y.TSS #define Y variable
    
    lr = linear_model.LinearRegression() #Call linear regression model
    lr.fit(X, Y_mreg) #Fit regression
    Y_pred = lr.predict(X) #Calculate points predicted to measure accuracy
    R2 = metrics.r2_score(Y_mreg, Y_pred) #Calculate R2 score
    print("MSE: {0} and R2: {1}".format(round(np.mean((Y_pred - Y_mreg)**2),4), R2.round(4)))
    print("")
    print("Coefficients: {0}".format(lr.coef_))
    print("")

#Perform simple logistic regression
log_reg = input("Perform a simple logistic regression (Y):")
if log_reg == 'Y' or log_reg == 'y':
    Y_lreg = Y.TSS #define Y variable
    
    lr = linear_model.LogisticRegression(multi_class='multinomial') #Call logistic regression model, BUT, normally, multinomial technique is not used to predict an ordered varialbe. Why do we use, instead? Look it up!
    lr.fit(X, Y_lreg) #Fit regression
    Y_pred = lr.predict(X) #Calculate points predicted to measure accuracy
    Accuracy = metrics.accuracy_score(Y_lreg, Y_pred) #Calculate accuracy score
    print("Accuracy: {0}".format(Accuracy.round(4)))
    print("")
    print("Coefficients: {0}".format(lr.coef_))


#Perform logistic regression with train_test_split
log_reg_tts = input("Perform a logistic regression with train_test_split (Y):")
if log_reg_tts == 'Y' or log_reg_tts == 'y':
    Y_lreg = Y_train.TSS #define Y variable
    
    lr = linear_model.LogisticRegression(multi_class='multinomial') #Call logistic regression model
    lr.fit(X_train, Y_lreg) #Fit regression
    Y_pred_train = lr.predict(X_train) #Calculate points predicted to measure accuracy
    Y_pred_test = lr.predict(X_test) #Calculate points predicted to measure accuracy
    Accuracy_train = metrics.accuracy_score(Y_lreg, Y_pred_train) #Calculate accuracy score
    Accuracy_test = metrics.accuracy_score(Y_test.TSS, Y_pred_test) #Calculate accuracy score
    print("")
    print("Training accuracy: {0} and testing accuracy: {1}".format(Accuracy_train.round(4), Accuracy_test.round(4)))
    print("")

#Perform logistic regression with train_test_split and cross validation
log_reg_tts_cv = input("Perform a logistic regression with train_test_split and cross-validation (Y):")
if log_reg_tts_cv == 'Y' or log_reg_tts_cv == 'y':
    Y_lreg = Y_train.TSS #define Y variable
    
    #Defining the model as a variable
    lr = linear_model.LogisticRegression(multi_class='multinomial') #Call logistic regression model
    
    #Cross Validation (CV) process
    scores = cross_val_score(lr, X_train, Y_lreg, cv=5) #actual cross-validation process
    print("")
    print(scores)
    print("Accuracy: {0} (+/- {1})".format(scores.mean().round(2), (scores.std() * 2).round(2)))
    print("")
    
    lr.fit(X_train, Y_lreg) #Fit regression
    Y_pred_train = lr.predict(X_train) #Calculate points predicted to measure accuracy
    Y_pred_test = lr.predict(X_test) #Calculate points predicted to measure accuracy
    Accuracy_train = metrics.accuracy_score(Y_lreg, Y_pred_train) #Calculate accuracy score
    Accuracy_test = metrics.accuracy_score(Y_test.TSS, Y_pred_test) #Calculate accuracy score
    print("Training accuracy: {0} and testing accuracy: {1}".format(Accuracy_train.round(4), Accuracy_test.round(4)))

#Goodbye message
print('')
print("Good Bye")
