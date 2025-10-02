import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ======= Configura estos valores =======
MODEL_PATH = "models/llama3_lora"
PROMPT = """Dame una canción inspirada en Enjambre, sobre la nostalgia y el paso del tiempo."""
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_P = 0.95

# ======================================
# Carga tokenizer y modelo con LoRA
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    quantization_config=bnb_config
)
model.eval()

# Tokeniza prompt
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

# Genera texto
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decodifica y muestra
generated = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n========= Canción generada =========\n")
print(generated[len(PROMPT):].strip())
print("\n====================================")
# Guarda en archivo
with open("results/LlaMA/generated_lyrics.txt", "w", encoding="utf-8") as f:
    f.write(generated)
