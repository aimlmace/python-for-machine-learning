import csv
data = []
with open('km.csv') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(row)

print(data)

kc1 = [float(data[0][1]),float(data[0][2])]
kc2 = [float(data[1][1]),float(data[1][2])]
kc3 = [float(data[2][1]),float(data[2][2])]

c1 = []
c2 = []
c3 = []
counter = 0

while True:
    counter += 1
    c1 = []
    c2 = []
    c3 = []

    for i in range(len(data)):
        x = float(data[i][1])
        y = float(data[i][2])

        d1 = (x-kc1[0])**2 + (y-kc1[1])**2
        d2 = (x-kc2[0])**2 + (y-kc2[1])**2
        d3 = (x-kc3[0])**2 + (y-kc3[1])**2

        if d1 < d2 and d1 < d3:
            c1.append([x,y])
        elif d2 < d1 and d2 < d3:
            c2.append([x,y])
        else:
            c3.append([x,y])
        
    kc1 = [0.0,0.0]
    kc2 = [0.0,0.0]
    kc3 = [0.0,0.0]

    for i in range(len(c1)):
        kc1[0] += c1[i][0]
        kc1[1] += c1[i][1]
    kc1[0] /= len(c1)
    kc1[1] /= len(c1)

    for i in range(len(c2)):
        kc2[0] += c2[i][0]
        kc2[1] += c2[i][1]
    kc2[0] /= len(c2)
    kc2[1] /= len(c2)

    for i in range(len(c3)):
        kc3[0] += c3[i][0]
        kc3[1] += c3[i][1]
    kc3[0] /= len(c3)
    kc3[1] /= len(c3)

    if counter == 100:
        break
print(c1,c2,c3)
print(kc1,kc2,kc3)


