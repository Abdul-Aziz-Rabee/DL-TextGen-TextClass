import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/llama3_v3")
    parser.add_argument("--prompt", type=str, default="Vuelvo a mirar el reloj,\n")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default="results/LlaMA")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Carga tokenizer y modelo LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
        quantization_config=bnb_config
    )
    model.eval()

    # Tokeniza prompt semilla
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)

    # Genera texto
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decodifica y recorta prompt
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_only = generated[len(args.prompt):].strip()

    # Imprime
    print("\n========= Canción generada =========\n")
    print(args.prompt + generated_only)
    print("\n====================================")

    # Guarda en archivo txt + json con parámetros
    fname = f"lyric_temp{args.temperature}_top{args.top_p}_max{args.max_new_tokens}.txt"
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(args.prompt + generated_only)

    meta = {
        "prompt": args.prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "model_path": args.model_path,
        "output_file": fname
    }
    with open(fpath.replace('.txt', '.json'), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
