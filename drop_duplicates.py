import pandas as pd
import numpy as np

n = int(input("Enter the dimension"))
array = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        array[i][j] = int(input())

df = pd.DataFrame(array)
print(df)
print()

df_no_dup = df.drop_duplicates(keep=False)
print(df_no_dup)
print()