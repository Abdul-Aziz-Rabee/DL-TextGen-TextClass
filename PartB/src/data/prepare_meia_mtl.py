import os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from preprocessing.limpieza import limpieza_basica
import json as jsonlib
from pathlib import Path

def preprocess_text(text, use_limpieza_basica=False, metodo="ftfy"):
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

def load_original_labels(name):
    path = Path("data/label_mappings") / f"{name}_labels_original.json"
    with open(path, "r", encoding="utf-8") as f:
        labels = jsonlib.load(f)
    mapping = {val: i for i, val in enumerate(labels)}
    return labels, mapping


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
    df["Review"] = df["Review"].apply(lambda x: preprocess_text(x, use_limpieza_basica=False))

    df["text"] = df["Review"].str.strip()

    label_mappings = {}

    # === POLARITY ===
    path_pol = Path("data/label_mappings/polarity_labels_original.json")
    with open(path_pol, "r", encoding="utf-8") as f:
        polarity_labels = jsonlib.load(f)

    polarity2id = {val: i for i, val in enumerate(polarity_labels)}
    id2polarity = {i: val for i, val in enumerate(polarity_labels)}

    df["polarity_label"] = df["Polarity"].apply(lambda x: polarity2id.get(x, -1))
    df = df[df["polarity_label"] != -1]

    label_mappings["polarity"] = {
        "label2id": polarity2id,
        "id2label": id2polarity
    }

    # === TYPE ===
    path_type = Path("data/label_mappings/type_labels_original.json")
    with open(path_type, "r", encoding="utf-8") as f:
        type_labels = jsonlib.load(f)

    type2id = {val: i for i, val in enumerate(type_labels)}
    id2type = {i: val for i, val in enumerate(type_labels)}

    df["type_label"] = df["Type"].apply(lambda x: type2id.get(x, -1))
    df = df[df["type_label"] != -1]

    label_mappings["type"] = {
        "label2id": type2id,
        "id2label": id2type
    }

    # === TOWN ===
    path_town = Path("data/label_mappings/town_labels_original.json")
    with open(path_town, "r", encoding="utf-8") as f:
        town_labels = jsonlib.load(f)

    town2id = {val: i for i, val in enumerate(town_labels)}
    id2town = {i: val for i, val in enumerate(town_labels)}

    df["town_label"] = df["Town"].apply(lambda x: town2id.get(x, -1))
    df = df[df["town_label"] != -1]

    label_mappings["town"] = {
        "label2id": town2id,
        "id2label": id2town
    }

    # === Guardar si quieres persistir el mapping actual ===
    with open("data/label_mappings/current_label_mappings.json", "w", encoding="utf-8") as f:
        jsonlib.dump(label_mappings, f, ensure_ascii=False, indent=2)

    print("✅ Mapeos completos con id2label generados.")

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
