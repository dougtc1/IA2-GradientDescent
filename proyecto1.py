from statistics import mean, pstdev
from csv import reader
import matplotlib.pyplot as plt
import numpy as np

def normalize(data):
	for attr in data.values():
		#print(sum(attr)/len(attr))
		media = mean(attr)
		desviacion = pstdev(attr,media)
		#
		##print(media)
		#print(desviacion)
		for x in range(0,len(data[0])):
			attr[x] = (attr[x] - media) / desviacion
	# media = mean(data[1])
	# desviacion = pstdev(data[1],media)
	# print(media)
	# print(desviacion)

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
		lista[x] = np.array(lista[x])
			
	return lista

def evaluar(thetas, valores):
	valor = thetas[0]

	for i in range(1, len(thetas)):
		valor += thetas[i]*valores[i - 1]

	return valor

def costoso(alpha, datos, iteraciones):
	numAtr = len(datos) 	# Cantidad atributos
	m = len(datos[0]) 		# Cantidad instancias
	thetasVector = np.zeros(numAtr)
	predictionVector = datos[numAtr - 1]
	costVector = []

	for i in range(iteraciones):
		temp = 0
		restas = [0] * numAtr
		for j in range(m):
			valores = []
			for k in range(numAtr - 1):
				valores.append(datos[k][j])
			for l in range(numAtr):
				x = evaluar(thetasVector, valores) - datos[numAtr - 1][j] 
				temp += x**2 	# Necesario para calcular bien el J(theta)
				if l != 0:
					x = x * datos[l - 1][j]
				restas[l] += x

		for p in range(numAtr):
			thetasVector[p] -= alpha * (1 / m) * restas[p]
		
		costVector.append((1 / (2 * m)) * temp) 

	return thetasVector, costVector

# "Main"
def main():
	#####################################
	# Actividad 1.1 - Archivo = x01.csv #
	#####################################
	
	numIteraciones = int(input("Ingrese el número de iteraciones para la primera actividad: "))
	alpha = 0.1
	lista = loadCsv("x01.csv")
	normalize(lista)
	cosaVector = []
	thetas, jota = (costoso(alpha, lista, numIteraciones))
	
	for i in range(len(lista[0])):
		cosa = thetas[0] + thetas[1]*lista[0][i]
		cosaVector.append(cosa)
		#print("Predicción: " + str(cosa) + " Valor real: " + str(lista[1][i]))
		#print("Predicción desnormalizada: " + str(cosa*pstdev(lista[0])+mean(lista[0])) + " Valor real desnormalizado: " + str(lista[1][i]*pstdev(lista[1])+mean(lista[1])) + "\n")
	print("Los thetas obtenidos en 1.1 son: ", thetas)
	print("El vector que minimiza los costos en 1.1 es el siguiente: ", cosaVector)
	print("\n")
	
	# J(theta) vs iteraciones
	plt.plot(jota)
	plt.title("Curva de convergencia")
	plt.ylabel("Costos")
	plt.xlabel("Numero de iteraciones")
	plt.show()
	
	# Scatter plot con curva que minimiza costos
	prueba = np.random.rand(len(lista[0])) # Colores distintos
	plt.title("Scatter plot con curva que minimiza costos")
	plt.scatter(lista[0],lista[1],c=prueba)
	plt.plot(cosaVector, cosaVector)
	plt.show()
	
	#####################################
	# Actividad 1.2 - Archivo = x08.csv #
	#####################################
	
	numIteraciones = int(input("Ingrese el número de iteraciones para la segunda actividad: "))
	lista.clear()
	lista = loadCsv("x08.csv")
	
	alphaList = [0.1,0.3,0.5,0.7,0.9,1.0]
	jotaList = [[],[],[],[],[],[]]
	thetaList = [[],[],[],[],[],[]]
	for i in range(len(alphaList)):
		thetaList[i],jotaList[i] = costoso(alphaList[i], lista, numIteraciones)

	# Parte a
	# J(theta) vs iteraciones
	plt.plot(jotaList[0])
	plt.title("Curva de convergencia")
	plt.ylabel("Costos")
	plt.xlabel("Numero de iteraciones")
	plt.show()
	
	# Parte b
	# Conjunto de curvas con distintos alphas
	plt.plot(jotaList[0])
	plt.plot(jotaList[1])
	plt.plot(jotaList[2])
	plt.plot(jotaList[3])
	plt.plot(jotaList[4])
	plt.plot(jotaList[5])
	plt.title("Curvas de convergencia")
	plt.ylabel("Costos")
	plt.xlabel("Numero de iteraciones")
	plt.show()
	
	# Parte c
	normalize(lista)
	jota = []
	thetas = []
	thetas,jota = costoso(alphaList[0], lista, numIteraciones)

	plt.plot(jota)
	plt.title("Curva de convergencia")
	plt.ylabel("Costos")
	plt.xlabel("Numero de iteraciones")
	plt.show()
	
	print("Los thetas obtenidos en 1.2 c) son: ", thetas)
	print("El vector de costos en 1.2 c)  es el siguiente: ", jota)
	print("\n")

	###########################################
	# Actividad 3 - Archivo = AmesHousing.txt #
	###########################################

if __name__ == '__main__':
	main()



"""def cost(alpha, thetasVector, valueVector,predictionVector,repeticiones):
	n = len(thetasVector)
	m = len(valueVector)
	restas = [0] * n
	#restar = 0
	#print(valueVector)
	#print(thetasVector)
	for i in range(0,repeticiones):
		restas = [0] * n
		restar = 0 
		restar1 = 0

		for y in range(m):

			# for j in range(n):
			# 	restast[j]+=

			restar += (thetasVector[0] + thetasVector[1] * valueVector[y]) - predictionVector[y]
			restar1 += ((thetasVector[0] + thetasVector[1] * valueVector[y]) - predictionVector[y])*valueVector[y]


		thetasVector[0] -= alpha * (1 / m) * restar
		thetasVector[1] -= alpha * (1 / m) * restar1

	#print(thetasVector) """


"""for i in range(len(lista[0])):
	cosa = thetas[0] + thetas[1]*lista[0][i] + thetas[2]*lista[1][i] + thetas[3]*lista[2][i]
	cosaVector.append(cosa)
	#print("Predicción: " + str(cosa) + " Valor real: " + str(lista[3][i]))
	#print("Predicción desnormalizada: " + str(cosa*pstdev(y)+mean(y)) + " Valor real desnormalizado: " + str(y[i]) + "\n")

#plt.plot(jota)
#plt.show()

####################
#MAIN Y TAL
#lista = thetaList[i],jotaList[i] = costoso(alphaList[i], lista, numIteraciones)
loadCsv('x08.csv')
#print(lista)
#print(tetas[1] > 0)
thetas = np.zeros(len(lista))
cost(alpha, thetas, lista[0], lista[1], numIteraciones)
# print(thetas)

	cosa = (thetas[0]+lista[0][i]*thetas[1])
	print('##########################################')
	print("Predicción: " + str(cosa*891.8 + 198.7) + " Valor real: " + str(lista[1][i]*922.74 + 283.13))
	print('VS')
	print('##########################################')

# #print(cosa)

##med = mean(lista[1])
#esv = pstdev(lista[1])
#print(cosa*desv+med)
#print(pstdev(lista[0]))
#plt.scatter(lista[0], lista[1])
#plt.show()
#print(lista)
#198.78998387096775
#891.8772590272449 """