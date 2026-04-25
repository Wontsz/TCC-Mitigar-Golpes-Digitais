import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./modelo_smishing_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def analisar(texto):
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predicao = torch.argmax(outputs.logits, dim=-1).item()
    return "GOLPE (1)" if predicao == 1 else "LEGÍTIMO (0)"

# Teste rápido
msg = input("Digite a mensagem para análise: ")
print(f"Resultado: {analisar(msg)}")