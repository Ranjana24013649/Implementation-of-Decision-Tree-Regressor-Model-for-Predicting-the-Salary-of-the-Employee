# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RANJANA R
RegisterNumber:  212224040270
*/
```
import pandas as pd

data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test, y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2
dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()

## Output:
Data.Head()

![319283190-964c85c1-4627-45c1-b905-0f77c3c3b12d](https://github.com/user-attachments/assets/200fd4c6-2c0a-4fba-8b5b-21654dffc736)


![319284497-b9d39342-e915-4ebd-906d-c7910f9bb566](https://github.com/user-attachments/assets/e9ce7152-1bdd-4f26-8e3a-ef7df72e3448)

![319284673-5cb9db1a-7819-42c7-be62-d137d6209c8c](https://github.com/user-attachments/assets/d76d57fc-ddb3-465a-9ef9-2866417b408b)

![319284919-acbff202-598d-4d01-a02f-7a7fb45741a0](https://github.com/user-attachments/assets/3272e6af-ad89-4e91-b2d6-bee51f288190)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
