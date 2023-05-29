from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Load dataset iris
df = pd.read_csv('VGG16_output/clf-data.csv')

# Pisahkan fitur dan target dari dataset
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Buat objek klasifikasi SVM
svm = svm.SVC()

# Latih model menggunakan data latih
svm.fit(X_train, y_train)

# Lakukan prediksi menggunakan data uji
y_pred = svm.predict(X_test)

# Hitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Hitung presisi
precision = precision_score(y_test, y_pred)
print("Precision: {:.2f}%".format(precision * 100))

# Hitung recall
recall = recall_score(y_test, y_pred)
print("Recall: {:.2f}%".format(recall * 100))

# Hitung f1-score
f1score = f1_score(y_test, y_pred)
print("F1-Score: {:.2f}%".format(f1score * 100))

# Hitung mean square error
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error (MSE): {:.2f}".format(mse))
