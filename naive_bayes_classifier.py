import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/home/cs-ai-25/exp-ml/Naive-Bayes-Classification-Data.csv')
X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
precision = precision_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.2f}')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Recall: {recall:.2f}')

glucose = input('Enter Glucose Level: ')
bloodpressure = input('Enter Blood Pressure Level: ')

user_input = {
    'glucose':glucose,
    'bloodpressure':bloodpressure
}

user_input_df = pd.DataFrame([user_input])

user_prediction = model.predict(user_input_df)

print(f'The Prediction for Diabetes is : {user_prediction[0]}')