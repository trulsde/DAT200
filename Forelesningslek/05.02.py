import pandas as pd

#1 - Name columns and rows:
iris = pd.read_csv("iris.csv", header=None)
irisDF = pd.DataFrame(iris)
num_entries = irisDF.shape[0]
new_cnames = {0: 'sepal_length', 1: 'sepal_width', 2: 'petal_length', 3: 'petal_width', 4: 'types'}
irisDF = irisDF.rename(columns=new_cnames)
flower_ind = [f'flower_{x + 1}' for x in range(num_entries)]

irisDF.index = flower_ind
print(irisDF.head())

#2 - Find unique values in column and compute mean:

# unique_classes = list(set(irisDF['types']))
unique_classes = irisDF['types'].unique()
print(unique_classes)

iris_pivot = irisDF.groupby('types')
class_means = iris_pivot.mean()
print(class_means)

#3 - Count occurances of each class, create subsets, view last 10 rows:

class_occurences = irisDF['types'].value_counts()
irisSetosaDF = irisDF[irisDF['types'] == 'Iris-setosa']
irisVersicolorDF = irisDF[irisDF['types'] == 'Iris-versicolor']
irisVirginicaDF = irisDF[irisDF['types'] == 'Iris-virginica']

last_10_instances = irisDF['sepal_length'].tail(10)
print(last_10_instances)
