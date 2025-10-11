import pandas as pd
import os 
basedir = "C:\\Users\\uzgre\\Codes\\Python\\DL Texto e Imagenes\\Tarea2-DL-TextGen-TextClass\\PartB\\data"
print(basedir)
ruta = os.path.join(basedir,'raw', 'MeIA_2025_train.csv')
print(ruta)
df = pd.read_csv(ruta)
print(df['Town'].value_counts().head(20))
print("Etiquetas únicas:", len(df['Town'].unique()))


