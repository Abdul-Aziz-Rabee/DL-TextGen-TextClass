import os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from preprocessing.limpieza import limpieza_basica


def preprocess_text(text, use_limpieza_basica=True, metodo="ftfy"):
    if not isinstance(text, str): 
        return ""
    try:
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Aplicar limpieza básica si se indica
    if use_limpieza_basica:
        text = limpieza_basica(text, metodo=metodo)    

    # Eliminar URLs y limpiar espacios
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_prepare_meia_for_mtl(
    data_file="MeIA_2025_train.csv",
    test_size=0.2,
    random_state=42
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "data/raw", data_file)

    df = pd.read_csv(data_path)
    print(f"✅ Datos cargados: {df.shape[0]} reseñas")

    # Limpieza y normalización
    df["Review"] = df["Review"].apply(lambda x: preprocess_text(x, use_limpieza_basica=True, metodo="ftfy"))

    df["text"] = df["Review"].str.strip()

    # Etiquetas
    label_mappings = {}
    df["polarity_label"] = df["Polarity"].astype(int) - 1
    label_mappings["polarity"] = {i: i+1 for i in range(5)}

    type_labels, type_categories = pd.factorize(df["Type"])
    df["type_label"] = type_labels
    label_mappings["type"] = {i: cat for i, cat in enumerate(type_categories)}

    town_labels, town_categories = pd.factorize(df["Town"])
    df["town_label"] = town_labels
    label_mappings["town"] = {i: cat for i, cat in enumerate(town_categories)}

    # División estratificada por Polarity
    final_cols = ["text", "polarity_label", "type_label", "town_label"]
    train_df, eval_df = train_test_split(
        df[final_cols],
        test_size=test_size,
        random_state=random_state,
        stratify=df["polarity_label"]
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    print(f"📊 Split completado: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    return {"train": train_dataset, "eval": eval_dataset, "label_mappings": label_mappings}

if __name__ == "__main__":
    data = load_and_prepare_meia_for_mtl()
    print("\nEjemplo:", data["train"][0])
    print("\nMapeos:", data["label_mappings"])
