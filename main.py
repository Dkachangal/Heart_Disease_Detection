import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')

# need to convert all the strings to int
x = df.drop('HeartDisease', axis = 1)
y = df['HeartDisease']
strCols = df[['Sex', 'ChestPainType','RestingECG', 'ExerciseAngina', 'ST_Slope']]

for col in strCols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# 1. LOGISTIC REGRESSION
LogisticMode = LogisticRegression()

LogisticMode.fit(x_train, y_train)

Logistic_y_predict = LogisticMode.predict(x_test)

accLogistic = accuracy_score(Logistic_y_predict, y_test)

print("Logistic Regression Accuracy:", str(accLogistic * 100) +  "%")


# 2. K N N
KnnModel = KNeighborsClassifier(n_neighbors=5)

KnnModel.fit(x_train, y_train)

Knn_y_predict = KnnModel.predict(x_test)

accKnn = accuracy_score(y_test, Knn_y_predict)
print("K-Nearest Neighbors Accuracy:", str(accKnn * 100) +  "%")

# 3. DECISION TREE
# 4. NAIVE BAYES

# sns.heatmap(x.corr(), annot=True)
# plt.show()