# DL-TextGen-TextClass

Deep learning project on practical implementation of text generation and text classification pipelines with PyTorch and Hugging Face using RNNs, LSTMs, GRUs, and Transformers.

Este repositorio contiene la **Tarea 2** del curso de maestría *Deep Learning para Procesamiento de Texto e Imágenes*.  

## 📂 Estructura del proyecto

```bash
DL-TextGen-TextClass/
├── PartA/   # Generación de letras de canciones
├── PartB/   # Clasificación de reseñas turísticas
├── requirements.txt
└── README.md
```


- `PartA/README_A.md`: instrucciones específicas para generación de texto.
- `PartB/README_B.md`: instrucciones específicas para clasificación.

## Cómo ejecutar
El proyecto está preparado para correrse tanto en local como en el clúster **Lab-SB (CIMAT)**.  
Ejemplo (Parte A, caracter-level RNN):
```bash
cd PartA
sbatch jobs/run_char.sh char_rnn_1
```

📦 Dependencias

- Python 3.11
- PyTorch + CUDA
- Transformers
- Datasets
- Accelerate
- Scikit-learn
- Matplotlib, Pandas, Numpy

Instalar con:
```bash
conda env create -f jobs/environment.yml
```
Resultados esperados

- Perplejidad (PPL) en generación.
- Accuracy y F1-score en clasificación.
- Gráficas de curvas de entrenamiento, matrices de confusión y ejemplos de letras generadas.
- Tablas comparativas de arquitecturas y tiempos de cómputo.

## Autor
Uziel Isaí Lujan López — M.Sc. in Statistical Computing at CIMAT

'uziel.lujan@cimat.mx'

[LinkedIn](https://www.linkedin.com/in/uziel-lujan/) | [GitHub](https://github.com/UzielLujan)