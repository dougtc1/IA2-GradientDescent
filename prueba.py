from pyexcel_ods3 import get_data

def aEntero(s):
    try: 
        return int(s)
    except ValueError:
        return s

def limpiar(data):
	#data = get_data("AmesHousing.ods")
	datos = {}
	for elem in (data['AmesHousing']):
		#print(elem)
		#Elimiar Order y PID
		elem = elem[2:]
		n = len(elem) - 1
		
		for i in range(len(elem)):
			if elem[n -1] != 'Normal' or elem[45] >  1500 or '' in elem:
				break
			try:
				datos[i].append(aEntero(elem[i]))
			except:
				datos[i] = []
				datos[i].append(aEntero(elem[i]))

	return datos
	
def main():
	data = get_data("AmesHousing.ods")
	datos = limpiar(data)
	#for elem in datos:
	#	print(elem)

if __name__ == '__main__':
	main()