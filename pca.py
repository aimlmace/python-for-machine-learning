import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, :4]
y = iris.target
print(pd.DataFrame(data=iris.data, columns=iris.feature_names))
pca = PCA(n_components=3)
col = pca.fit_transform(X)
df = pd.DataFrame(col, columns=['PC1', 'PC2','PC3'])
df['target'] = y
for i in range(3):
    plt.scatter(df.loc[y==i, 'PC1'], df.loc[y==i, 'PC2'], label=iris.target_names[i])
print(df)
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_Width')
plt.title('PCA of Iris Dataset')
plt.legend(title='Species')
plt.show()