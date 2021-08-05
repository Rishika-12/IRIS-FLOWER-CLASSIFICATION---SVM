# IRIS-FLOWER-CLASSIFICATION---SVM
#IMPORTING NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline 
#IMPORTING THE DATASET
df=pd.read_csv("/content/iris dataset.csv")
df.head()
#CHECKING THE DATA AND CLEANING
df.isnull().sum()
df=df.dropna()
df.isnull().sum()
df.shape
a=df.dtypes[df.dtypes=="object"]
a
df=df.drop_duplicates()
df.shape
#VISUALISING THE DATA
sb.pairplot(df, hue="class")
#SEPERATING X AND Y
x=df.iloc[:,df.columns!="class"]
y=df.iloc[:,df.columns=="class"]
x
y
#SPLITTING TEST AND TRAIN DATA
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
xtrain
ytrain
xtest
ytest
#SVM ALGORITHM
from sklearn import svm
model=svm.SVC(kernel="linear")
model.fit(xtrain,ytrain)
model_output=model.predict(xtest)
model_output
#CHECKING ACCURACY
from sklearn.metrics import accuracy_score
acc1=accuracy_score(model_output,ytest)
acc1
#PREDICTING OR TESTING THE MODEL BY GIVING SAMPLE VALUES
predic1=model.predict([[5.2,3.6,1.2,0.1]])
predic2=model.predict([[6,2.3,4,1.2]])
predic3=model.predict([[5.9,3,5.6,1.8]])
print(predic1)
print("\n")
print(predic2)
print("\n")
print(predic3)
