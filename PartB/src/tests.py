import pandas as pd
import os 
basedir = "C:\\Users\\uzgre\\Codes\\Python\\DL Texto e Imagenes\\Tarea2-DL-TextGen-TextClass\\PartB\\data"
#basedir = "C:\\Users\\uzgre\\Codes\\Python\\Projects\\LLMs-sentiment-analysis-mx\\data"
ruta = os.path.join(basedir,'raw', 'MeIA_2025_train.csv')
#ruta = os.path.join(basedir, 'Rest-Mex_2025_train.csv')
print(ruta)
df = pd.read_csv(ruta)
_, town_mapping_meia = pd.factorize(df["Type"])
print(town_mapping_meia)