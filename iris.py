import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# load the data file
iris_df = pd.read_csv("iris.csv")

# print the data rows 
print(iris_df.head())

# drop irrelevant columns 
iris_df.drop(columns = ['Id'], axis = 1, inplace = True)

# check for duplicates
print(iris_df.duplicated().sum())

# drop all duplicates
iris_df.drop_duplicates(inplace = True)

# recheck for duplicates 
print(iris_df.duplicated().sum())

# check for missing data in each columns 
print(iris_df.isnull().sum())

# split the data into 80% training and 20% testing 
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop(columns = ['Species'], axis = 1), 
                                                    iris_df['Species'], 
                                                    test_size = 0.2, 
                                                    random_state = 42, 
                                                    stratify = iris_df['Species'])

# scale the features
scaler = StandardScaler()
X_train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(X_train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
X_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.transform(X_test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
print(X_train.head())

# fit the random forest classifier
rfc = RandomForestClassifier(n_estimators = 150, random_state = 42)
rfc.fit(X_train, y_train)

# evaluate the model
y_pred = rfc.predict(X_test)

# display the classification report
report = classification_report(y_test, y_pred)
print(report)

with open("classification_report.txt", "w") as f:
    f.write("## Classification Report\n\n'''text\n")
    f.write(report)
    f.write("\n'''\n")

# save confusion matrix as image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
fig, ax = plt.subplots(figsize = (6, 6))
disp.plot(cmap = "Blues", ax = ax, colorbar = False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi = 120)
