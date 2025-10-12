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

# scripts/plot_f1_comparison.py
import matplotlib.pyplot as plt

# === Datos ===
models = ["RNN", "LSTM", "GRU", "CNN", "BETO MTL SO + LoRA"]
f1_weighted = [0.462, 0.481, 0.459, 0.467, 0.683]

# === Configuración estética ===
plt.figure(figsize=(8, 4))
bars = plt.barh(models, f1_weighted, color="#2b83ba", edgecolor="black")

# Resaltar BETO
bars[-1].set_color("#fdae61")
bars[-1].set_edgecolor("black")

# Etiquetas
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", va='center', fontsize=10)

plt.xlabel("F1 ponderado")
plt.title("Comparación de desempeño en F1 Weighted\nModelos clásicos vs BETO MTL SO + LoRA")
plt.xlim(0.4, 0.8)
plt.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("Comparativa_F1_BETO_vs_clasicos.png", dpi=300)
plt.show()
