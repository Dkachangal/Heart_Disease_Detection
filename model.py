import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import joblib

df = pd.read_csv('heart.csv')
scaler = StandardScaler()

cholesMedian = df[df['Cholesterol'] > 0]['Cholesterol'].median()
df['Cholesterol'] = df['Cholesterol'].replace(0, cholesMedian)

x = df.drop(['HeartDisease', 'Sex', 'ChestPainType','RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1)
x_num = df.drop(['HeartDisease', 'Sex', 'ChestPainType','RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1)

y = df['HeartDisease']
strCols = ['Sex', 'ChestPainType','RestingECG', 'ExerciseAngina', 'ST_Slope']

# need to convert all the strings to int
ohe = OneHotEncoder(sparse_output=False, drop='first')

strColArr = ohe.fit_transform(df[strCols])
strColDF = pd.DataFrame(strColArr, columns=ohe.get_feature_names_out(strCols))
x = pd.concat([x_num, strColDF], axis=1)

print(x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 1. LOGISTIC REGRESSION
LogisticMode = LogisticRegression(max_iter=10000)

LogisticMode.fit(x_train, y_train)

Logistic_y_predict = LogisticMode.predict(x_test)

accLogistic = accuracy_score(Logistic_y_predict, y_test)

print("Logistic Regression Accuracy:", str(accLogistic * 100) +  "%")


# # 2. K N N
KnnModel = KNeighborsClassifier(n_neighbors=5)

KnnModel.fit(x_train, y_train)

Knn_y_predict = KnnModel.predict(x_test)

accKnn = accuracy_score(y_test, Knn_y_predict)
print("K-Nearest Neighbors Accuracy:", str(accKnn * 100) +  "%")


# 3. NAIVE BAYES
NaiveBayesModel = GaussianNB()

NaiveBayesModel.fit(x_train, y_train)

NaiveBayes_y_predict = NaiveBayesModel.predict(x_test)

accNaive = accuracy_score(y_test, NaiveBayes_y_predict)
print("Naive Bayes Accuracy:", str(accNaive * 100) +  "%")


# 4. DECISION TREE
DecisionTreeModel = DecisionTreeClassifier(random_state = 42, max_depth = 4)

DecisionTreeModel.fit(x_train, y_train)

DecisionTree_y_predict = DecisionTreeModel.predict(x_test)

accuDecisionTree = accuracy_score(DecisionTree_y_predict, y_test)
print("Decision Tree Accuracy:", str(accuDecisionTree * 100) +  "%")

# Creating pickle files of Logistic Regression
print(Logistic_y_predict)
joblib.dump(LogisticMode, 'LogisticModelFile.pkl')
joblib.dump(x, 'xValues.pkl')
joblib.dump(scaler, 'scaler.pkl')

