Hola, te proporciono el contexto para trabajar el dataset `MeIA_2025_train.csv` para la parte B de nuestro proyecto. Es un resumen de los hallazgos del análisis exploratorio previo sobre el dataset `MeIA_2025_train.csv`:

**1. Estructura del Dataset:**
* El corpus contiene las columnas: `Review`, `Polarity`, `Type`, `Town` y `Region`.
* La columna objetivo principal es `Polarity`. No hay valores nulos.

```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 5 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Review    5000 non-null   object 
 1   Polarity  5000 non-null   float64
 2   Town      5000 non-null   object 
 3   Region    5000 non-null   object 
 4   Type      5000 non-null   object 
dtypes: float64(1), object(4)
memory usage: 195.4+ KB
```

Breve analisis de las variables objetivo:
```bash
Polarity
5.0    1200
4.0    1100
3.0    1000
2.0     900
1.0     800
Name: count, dtype: int64
```
```bash
Type
Restaurant    2037
Hotel         1511
Attractive    1452
Name: count, dtype: int64
```
```bash 
Town
Tulum                         961
Isla_Mujeres                  646
San_Cristobal_de_las_Casas    321
Valladolid                    269
Bacalar                       250
Palenque                      222
Sayulita                      207
Valle_de_Bravo                153
Tlaquepaque                   147
Taxco                         118
Tequisquiapan                 114
Patzcuaro                     110
Loreto                        110
Tepoztlan                     107
Teotihuacan                   104
Ajijic                        102
TodosSantos                    99
Metepec                        77
Orizaba                        74
Tequila                        74
Cholula                        73
Ixtapan_de_la_Sal              65
Bernal                         58
Huasca_de_Ocampo               51
Creel                          50
Atlixco                        48
Izamal                         47
Tepotzotlan                    39
Zacatlan                       36
Parras                         30
Mazunte                        30
Chiapa_de_Corzo                30
Dolores_Hidalgo                29
Cuatro_Cienegas                26
Cuetzalan                      25
Xilitla                        24
Malinalco                      24
Tapalpa                        21
Coatepec                       16
Real_de_Catorce                13
Name: count, dtype: int64
```

**2. Hallazgos Clave de la Limpieza de Texto:**
* **Problemas de Codificación (Mojibake):** El texto original contenía artefactos de mala codificación (ej. `Ã¡` en lugar de `á`). Esto se solucionó usando la librería `ftfy`.
* **Reseñas Truncadas:** Se detectaron y eliminaron patrones de scraping como `...Más` al final de algunas reseñas.
* Estos pasos de limpieza básica están consolidados en una función `limpieza_basica` dentro del módulo `limpieza.py`.

**3. Características de las Variables:**
* **`Polarity`:** Originalmente era `float` (1.0, 2.0, etc.) y se transformó a `int` para ser usada como clase categórica. Su distribución está ligeramente desbalanceada.
* **`Town`:** Es una variable con alta cardinalidad y un fuerte desbalance de clases. Pocas ciudades (como Tulum) concentran una gran cantidad de las reseñas.
* **`custom_words`:** Durante el análisis, se identificó un conjunto de palabras muy frecuentes pero poco informativas para la polaridad (ej. 'hotel', 'restaurante'). Estas se filtran en etapas posteriores del pipeline.

Ahora, te proporcionaré los dos módulos de Python (`limpieza.py` y `main.py`) que implementan esta lógica y orquestan el pipeline completo.