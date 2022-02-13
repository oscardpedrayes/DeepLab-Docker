import numpy as np
import csv
import sys


# Fichero temporal generado por mpstat
datos = sys.argv[1] #'../../temp_cpu.csv'

#Fichero con los datos formateados
final = sys.argv[2] #"cpu_rendimiento.csv"

# Cabeceras del fichero final
with open(final, "w+") as f:
    f.write("Uso CPU[%] por core;\n")
    f.write("Fecha;all;0;1;2;3;4;5;6;7;\n")



fila = ""

# Leer los datos del fichero temporal y formatearlos
with open(datos, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if i > 0 and len(line) > 0:
                        
            linea = '{}'.format(line)
            items = linea.split(" ")
            
            if items[len(items)-1].replace("]", "").replace("'","").replace(",", ".") != "%idle":
                core = items[7]
                
                # Formatear los datos
                fecha = items[0].replace("[", "").replace("'","")
                uso_cpu = np.round(100 - float(items[len(items)-1].replace("]", "").replace("'","").replace(",", ".")), 2)
                
                # Fila a escribir
                fila = fila + ";" + str(uso_cpu)
                
                # Escribir cuando se tienen los datos de todos los cores
                if core == "7":
                    fila = fecha + fila
                    fila = fila + ";\n"
                    with open(final, "a+") as file:
                        file.write(fila)
                    fila = ""   