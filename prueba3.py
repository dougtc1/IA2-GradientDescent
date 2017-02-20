from pyexcel_ods3 import get_data
from statistics import mean, pstdev
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import copy

def aEntero(s):
    try: 
        return int(s)
    except ValueError:
        return s

def limpiar(data):
	#data = get_data("AmesHousing.ods")
	datos = {}
	iterElem = iter(data['AmesHousing'])
	next(iterElem)
	for elem in (iterElem):
		#Eliminar Order y PID
		elem = elem[2:]
		n = len(elem) - 1
		#print(elem)
		#exit(0)
		
		for i in range(len(elem)):
			if elem[n -1] != 'Normal' or elem[45] >  1500 or '' in elem:
				break
			try:
				datos[i].append(aEntero(elem[i]))
			except:
				datos[i] = []
				datos[i].append(aEntero(elem[i]))

	#######################
	#Convertir las vainas no n'umericas a n'umericas
	#######################
	for i in range (len(datos)):
		try:
			temp = int(datos[i][0])
		except:
			noNum = []
			for j in range(len(datos[i])):
				if datos[i][j] in noNum:
					datos[i][j] = noNum.index(datos[i][j])
				else:
					noNum.append(datos[i][j])
					datos[i][j] = noNum.index(datos[i][j])

	return datos
	
def normalize(data):
	for attr in data.values():
		#print(sum(attr)/len(attr))
		media = mean(attr)
		desviacion = pstdev(attr,media)
		#print(media)
		#print(desviacion)
		#print(attr)
		for x in range(0,len(data[0])):
			attr[x] = (attr[x] - media) / desviacion

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

def main():
	data = get_data("AmesHousing.ods")
	datos = limpiar(data)
	numAtributos = len(datos)
	cantInstancias = len(datos[0])
	#print(datos[n-2])
	datos[numAtributos-2] = datos.pop(numAtributos-1)
	numAtributos = len(datos)
	valorRealDesnormalizado = copy.deepcopy(datos[numAtributos-1])
	mediaReal = mean(valorRealDesnormalizado)
	desvReal = pstdev(valorRealDesnormalizado, mediaReal)

	"""	
	listaMedias = []
	listaDesvst = []

	for i in range(numAtributos):
		dummy = mean(datos[i])
		listaMedias.append(dummy)
		listaDesvst.append(pstdev(datos[i], dummy)) """

	# Normalizacion de los datos
	normalize(datos)

	# Toma aleatoria del 80%
	aleatorio = [i for i in range(cantInstancias)]
	shuffle(aleatorio)
	tamTrainingSet = int(round((cantInstancias * 80) / 100,0))
	tamTestSet = cantInstancias - tamTrainingSet
	trainingSet = {}
	testSet = {}
	realTestSet = []
	realTrainingSet = []
	for i in range(numAtributos):
		for j in range(cantInstancias):
			if (j < tamTrainingSet):
				try:
					trainingSet[i].append(datos[i][aleatorio[j]])
				except:
					trainingSet[i] = []
					trainingSet[i].append(datos[i][aleatorio[j]])
				realTrainingSet.append(valorRealDesnormalizado[aleatorio[j]])
			else:
				try:
					testSet[i].append(datos[i][aleatorio[j]])
				except:
					testSet[i] = []
					testSet[i].append(datos[i][aleatorio[j]])
				realTestSet.append(valorRealDesnormalizado[aleatorio[j]])

	#################
	# Entrenamiento #
	#################
	
	alpha = 0.1
	numIteraciones = int(input("Ingrese el número de iteraciones para la tercera actividad: "))
	thetas, jota = (costoso(alpha, trainingSet, numIteraciones))
	
	#print(testSet[numAtributos-1][tamTestSet - 1])
	#exit()

	#Evaluación con el training set
	mediaTrainingSet = mean(realTrainingSet)
	desvTrainingSet = pstdev(realTrainingSet,mediaTrainingSet)

	prediccionesTraining = []
	valoresRealesTraining = []
	for i in range(tamTrainingSet):
		instancia = []
		valoresRealesTraining.append(trainingSet[numAtributos - 1][i])
		for j in range(numAtributos - 1):
			instancia.append(trainingSet[j][i])
		prediccionesTraining.append((evaluar(thetas, instancia)*desvTrainingSet)+mediaTrainingSet)

	# Calculo de métricas
	temp = 0
	biasTrain = 0
	maxDevTrain = 0
	meanAbsDevTrain = 0
	MSETrain = 0
	for i in range(len(prediccionesTraining)):
		#print(i)
		temp = prediccionesTraining[i] - realTrainingSet[i] #valoresRealesTraining[i]
		biasTrain += temp
		maxDevTrain = max(maxDevTrain, abs(temp))
		meanAbsDevTrain += abs(temp)
		MSETrain += temp**2

	biasTrain /= len(prediccionesTraining)
	meanAbsDevTrain /= len(prediccionesTraining)
	MSETrain /= len(prediccionesTraining)

	#Evaluación con el test set
	mediaTestSet = mean(realTestSet)
	desvTestSet = pstdev(realTestSet,mediaTestSet)
	prediccionesTest = []
	valoresRealesTest = []
	for i in range(tamTestSet):
		instancia = []
		valoresRealesTest.append(testSet[numAtributos - 1][i])
		for j in range(numAtributos - 1):
			instancia.append(testSet[j][i])
		prediccionesTest.append((evaluar(thetas, instancia)*desvTestSet)+mediaTestSet)

	# Calculo de métricas

	temp = 0
	biasTest = 0
	maxDevTest = 0
	meanAbsDevTest = 0
	MSETest = 0
	for i in range(len(prediccionesTest)):
		temp = prediccionesTest[i] - realTestSet[i] #valoresRealesTest[i]
		biasTest += temp
		maxDevTest = max(maxDevTest, abs(temp))
		meanAbsDevTest += abs(temp)
		MSETest += temp**2

	biasTest /= len(prediccionesTest)
	meanAbsDevTest /= len(prediccionesTest)
	MSETest /= len(prediccionesTest)
	
	# print(bias)	
	# print(maxDev)
	# print(meanAbsDev)
	# print(MSE)
	# #Evaluación con el test set
	###################################################

		#Evaluaci'on con el training set
	# print(bias)	
	# print(maxDev)
	# print(meanAbsDev)
	# print(MSE)


	#print(valoresRealesTest[tamTestSet - 1])
	print(biasTrain)
	print(meanAbsDevTrain)
	print(maxDevTrain)
	print(MSETrain)
	"""
	# Plot de J(theta) vs iteraciones
	plt.plot(jota)
	plt.title("Curva de convergencia")
	plt.ylabel("Costos")
	plt.xlabel("Numero de iteraciones")
	plt.show()
	"""
	# Plot de normalizados Train
	#plt.figure(1)

	# subplot de Bias
	#plt.subplot(221)
	a = plt.bar(np.arange(1),biasTrain,0.35,alpha=0.8,color="b",label="Train")
	b = plt.bar(np.arange(1)+0.35,biasTest,0.35,alpha=0.8,color="g",label="Test")
	plt.xticks(np.arange(12)+0.35,('Train', 'Test'))
	#plt.yscale('linear')
	plt.title("Bias a")
	plt.grid(True)
	plt.tight_layout()
	plt.show()
	
	"""
	# subplot de mean absolute error
	plt.subplot(222)
	#plt.plot([meanAbsDevTrain, meanAbsDevTest])
	#plt.yscale('linear')
	plt.title("bias b Mean Absolute Error")
	plt.grid(True)

	# subplot de maximun deviation
	plt.subplot(223)
	plt.plot(maxDevTrain,maxDevTest)
	plt.yscale('linear')
	plt.title("Maximun Deviation")
	plt.grid(True)

	# subplot de mean square error
	plt.subplot(224)
	plt.plot(MSETrain,MSETest)
	plt.yscale('linear')
	plt.title("Mean Square Error")
	plt.grid(True)	
	
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
	plt.show()



	# Plot de normalizados Test
	plt.figure(1)

	# subplot de Bias
	plt.subplot(221)
	plt.plot(biasTrain, [range(-1,1)])
	plt.yscale('linear')
	plt.title("Bias")
	plt.grid(True)

	# subplot de mean absolute error
	plt.subplot(222)
	plt.plot(meanAbsDevTrain)
	plt.yscale('linear')
	plt.title("Mean Absolute Error")
	plt.grid(True)

	# subplot de maximun deviation
	plt.subplot(223)
	plt.plot(maxDevTrain)
	plt.yscale('linear')
	plt.title("Maximun Deviation")
	plt.grid(True)

	# subplot de mean square error
	plt.subplot(224)
	plt.plot(maxDevTrain)
	plt.yscale('linear')
	plt.title("Mean Square Error")
	plt.grid(True)	
	plt.show()


	#trainingSet.sort()
	#testSet.sort()
	#print("indices training: ",trainingSet)
	#print("indices test: ",testSet)
	#print(aleatorio)
	#for elem in datos:
	#	print(datos[elem])
	#print(len(datos)) """
if __name__ == '__main__':
	main()