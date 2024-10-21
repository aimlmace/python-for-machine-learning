import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
iris = datasets.load_iris()
X = iris.data[:, 1:3]  
y = iris.target         
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
def plot_decision_boundaries(X, y, model, title="SVM Decision Boundaries"):
    h = 0.02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    markers = ['o', 's', '^']
    colors = ['red', 'green', 'blue']
    class_names = iris.target_names
    for idx, label in enumerate(np.unique(y)):
        plt.scatter(X[y == label, 0], X[y == label, 1], 
                    c=colors[idx], marker=markers[idx], 
                    edgecolors='k', s=100, label=class_names[label])
    plt.legend(loc='upper left', title='Classes')
    plt.xlabel('Sepal Width (Standardized)')
    plt.ylabel('Petal Length (Standardized)')
    plt.title(title)
    plt.show()
plot_decision_boundaries(X_train, y_train, svm_clf)
