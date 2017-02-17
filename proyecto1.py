from statistics import mean, pstdev
from csv import reader
from matplotlib import pyplot as plt
import numpy as np

def normalize(data):
	for attr in data.values():
		media = mean(attr)
		desviacion = pstdev(attr,media)
		#print(media)
		#print(desviacion)
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
		print('eu')
		lista[x] = np.array(lista[x])
			
	return lista

def cost(alpha, thetasVector, valueVector,predictionVector,repeticiones):
	n = len(thetasVector)
	#restar = 0
	#print(valueVector)
	#print(thetasVector)
	for i in range(0,repeticiones):
		for x in range(0,n):
			restar = 0 
			for y in range(len(valueVector)):
				if x == 1:
					restar += ((thetasVector[0] + thetasVector[1] * valueVector[y]) - predictionVector[y])*valueVector[y]
				else:
					restar += (thetasVector[0] + thetasVector[1] * valueVector[y]) - predictionVector[y]
			#print(restar)
			#print(x)
			#print((restar*0.1)/alpha)
			thetasVector[x] -= alpha * (1 / n) * restar
	#print(thetasVector)


# "Main"
numIteraciones = int(input("Ingrese el número de iteraciones: "))
alpha = 0.1
lista = loadCsv('x01.csv')
normalize(lista)
#print(lista)
#print(lista)
thetas = np.zeros(len(lista))
cost(alpha, thetas, lista[0], lista[1], numIteraciones)
#print(thetas)
cosa = (thetas[0]+1.620*thetas[1])
print(cosa)
##med = mean(lista[1])
#esv = pstdev(lista[1])
#print(cosa*desv+med)
#print(pstdev(lista[0]))
#plt.scatter(lista[0], lista[1])
#plt.show()
#print(lista)
#198.78998387096775
#891.8772590272449

