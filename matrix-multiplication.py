import numpy as np

r1 , c1= map(int,input("Enter no. rows and columns of first matrix").split())
r2 , c2= map(int,input("Enter no. rows and columns of second matrix").split())
array1 = np.ndarray((r1,c1))
array2 = np.ndarray((r2,c2))

if c1 != r2:
    raise ValueError("Matrices cannot be Multiplied")
else:
    print("Enter elements of first matrix")
    for i in range(r1):
        for j in range(c1):
            array1[i,j]=int(input())
    
    print("Enter elements of second matrix")
    for i in range(r2):
        for j in range(c2):
            array2[i,j]=int(input())
    result = array1.dot(array2)
    print(result)
    