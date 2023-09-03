import sys
import pickle
import pandas as pd

with open('modelo_logist.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

print(modelo_cargado)

if len(sys.argv)>0:

    arguments = sys.argv[1:]

    for arg in arguments:
        print(arg)
    
    # Divide cada argumento en una lista de valores usando ","
    argument_values = [arg.split(',') for arg in arguments]

    # Crea un DataFrame de Pandas con los valores divididos
    datos_nuevos = pd.DataFrame(argument_values, columns=["EK", "Skewness"])
    predicciones = modelo_cargado.predict(datos_nuevos)
    print(predicciones)

