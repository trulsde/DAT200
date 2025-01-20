from numpy import random as rd

x = rd.randint(100, size=(10, 2, 3))
print(x)
y = x[:4]
print(y)

for i in range(8, 9):
    print(i)

    empty_list = []
    empty_list.append(2)
