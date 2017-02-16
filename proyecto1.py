from statistics import mean, pstdev
from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def normalize(data):
	for attr in data.values():
		media = mean(attr)
		desviacion = pstdev(attr,media)
		for x in range(0,len(attr)):
			attr[x] = (attr[x] - media) / desviacion

def loadCsv(filename):
	lista = {}
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue

			for i in range(0, len(row)):
				try:
					lista[i].append(float(row[i]))
				except:
					lista[i] = []
					lista[i].append(float(row[i]))

	for x in range(0,len(lista)):
		lista[i] = np.array(lista[i])
			
	return lista

def cost(alpha, thetasVector, valueVector,predictionVector):
	n = len(thetasVector)

	for i in range(0,25):
		for x in range(0,n):
			thetasVector[x] = thetasVector[x] - alpha * (1 / n) * sum(((thetasVector * valueVector) - predictionVector[x])*valueVector[x])

	print(thetasVector)

# "Main"
alpha = 0.1
lista = loadCsv('x01.csv')
print(lista)
normalize(lista)
thetas = np.zeros(len(lista[0]))
cost(alpha, thetas, lista[0], lista[1])
#print(pstdev(lista[0]))
#plt.scatter(lista[0], lista[1])
#plt.show()
#print(lista)
