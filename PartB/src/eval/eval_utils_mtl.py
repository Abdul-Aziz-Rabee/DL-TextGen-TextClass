# src/eval_utils_mtl.py
import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_and_save_mtl_metrics(
    predictions, 
    labels, 
    run_name, 
    label_mappings,
    results_dir="results"
):
    """
    Calcula, guarda e imprime las métricas de evaluación y las matrices de confusión 
    para un modelo de aprendizaje multitarea, incluyendo el "Score" final de la competencia.

    Args:
        predictions (tuple): logits por tarea (polarity, type, town)
        labels (tuple): etiquetas verdaderas por tarea
        run_name (str): nombre de la ejecución
        label_mappings (dict): diccionario con los mapeos label2id e id2label
        results_dir (str): carpeta de salida
    """
    os.makedirs(results_dir, exist_ok=True)

    logits_polarity, logits_type, logits_town = predictions
    labels_polarity, labels_type, labels_town = labels

    preds_polarity = np.argmax(logits_polarity, axis=1)
    preds_type = np.argmax(logits_type, axis=1)
    preds_town = np.argmax(logits_town, axis=1)

    # --- Métricas por tarea ---
    f1_polarity = f1_score(labels_polarity, preds_polarity, average="weighted")
    f1_type = f1_score(labels_type, preds_type, average="weighted")
    f1_town = f1_score(labels_town, preds_town, average="weighted")

    final_score = (2 * f1_polarity + 1 * f1_type + 3 * f1_town) / 6.0
    
    metrics = {
        "Official_Score": final_score,
        "polarity_weighted_f1": f1_polarity,
        "type_weighted_f1": f1_type,
        "town_weighted_f1": f1_town,
        "polarity_accuracy": accuracy_score(labels_polarity, preds_polarity),
        "type_accuracy": accuracy_score(labels_type, preds_type),
        "town_accuracy": accuracy_score(labels_town, preds_town),
        "polarity_per_class_f1": f1_score(labels_polarity, preds_polarity, average=None).tolist(),
        "type_per_class_f1": f1_score(labels_type, preds_type, average=None).tolist(),
        "town_per_class_f1": f1_score(labels_town, preds_town, average=None).tolist(),
    }

    print(f"\n--- Métricas Finales para la ejecución: {run_name} ---")
    print(f"  Score Oficial (2-1-3): {metrics['Official_Score']:.4f}")
    print("  ------------------------------------")
    print(f"  F1 Polarity: {metrics['polarity_weighted_f1']:.4f}")
    print(f"  F1 Type:     {metrics['type_weighted_f1']:.4f}")
    print(f"  F1 Town:     {metrics['town_weighted_f1']:.4f}")

    # Guardar métricas en JSON
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(metrics_path, 'w', encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\nMétricas guardadas en: {metrics_path}")

    # --- Obtener nombres de etiquetas (id2label) ---
    def get_labels(mapping_dict):
        # Si viene en formato nuevo con id2label
        if isinstance(mapping_dict, dict) and "id2label" in mapping_dict:
            id2label = mapping_dict["id2label"]
            # ordenar por id
            sorted_labels = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in sorted(map(int, id2label.keys()))]
            return sorted_labels
        # Si viene en formato antiguo (label2id simple)
        elif isinstance(mapping_dict, dict):
            return list(mapping_dict.values())
        # Si es lista simple
        elif isinstance(mapping_dict, list):
            return mapping_dict
        else:
            return []

    labels_polarity_names = get_labels(label_mappings["polarity"])
    labels_type_names = get_labels(label_mappings["type"])
    labels_town_names = get_labels(label_mappings["town"])

    # --- Generar matrices de confusión con nombres legibles ---
    task_info = {
        "polarity": (labels_polarity, preds_polarity, labels_polarity_names),
        "type": (labels_type, preds_type, labels_type_names),
        "town": (labels_town, preds_town, labels_town_names)
    }

    for task_name, (y_true, y_pred, display_labels) in task_info.items():
        cm = confusion_matrix(y_true, y_pred)
        figsize = (20, 18) if task_name == "town" else (10, 8)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=display_labels, yticklabels=display_labels)
        plt.title(f"Matriz de Confusión - {task_name.capitalize()} - {run_name}")
        plt.ylabel("Etiqueta Verdadera")
        plt.xlabel("Etiqueta Predicha")
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"{run_name}_{task_name}_confusion_matrix.png")
        plt.savefig(plot_path)
        print(f"Matriz de confusión para '{task_name}' guardada en: {plot_path}")
        plt.close()
