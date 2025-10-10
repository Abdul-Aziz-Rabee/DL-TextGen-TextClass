import pandas as pd
df = pd.read_csv("data/raw/MeIA_2025_train.csv")
df["label"] = df["Polarity"].astype(int) - 1
print(df["label"].value_counts())
print("Etiquetas únicas:", sorted(df["label"].unique()))

