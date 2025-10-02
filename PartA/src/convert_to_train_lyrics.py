import os
import json

def convertir_canciones_a_jsonl(src_path, dest_path):
    with open(src_path, encoding="utf-8") as f:
        txt = f.read()
    canciones = [c.strip() for c in txt.split("<|startsong|>") if c.strip()]
    canciones = [c.replace("<|endsong|>", "").strip() for c in canciones]
    with open(dest_path, "w", encoding="utf-8") as f:
        for c in canciones:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    print(f"Archivo JSONL generado: {dest_path} ({len(canciones)} canciones)")


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH = os.path.join(base_path, "data", "canciones_clean.txt")
    PATH_OUT = os.path.join(base_path, "data", "train_lyrics.json")
    convertir_canciones_a_jsonl(PATH, PATH_OUT)


