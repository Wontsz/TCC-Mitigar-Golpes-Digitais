import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "Wontsimt/detector-phishing-v4-links" #nuvem, detector-phishing-v1 para modelo de 90%
#MODEL_PATH = "./modelo_smishing_final" #local

print("Carregando o modelo diretamente da nuvem...")
MODEL_PATH = "./modelo_smishing_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# CONFIGURAÇÃO DE SENSIBILIDADE
THRESHOLD = 0.3

def analisar_mensagem(texto):
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        prob_golpe = probs[0][1].item()
    
    status = "GOLPE (1)" if prob_golpe >= THRESHOLD else "LEGÍTIMO (0)"
    
    print(f"\nTexto: {texto}")
    print(f"Probabilidade de Golpe: {prob_golpe:.2%}")
    print(f"Decisão (Threshold {THRESHOLD}): {status}")

# Loop de teste
print("\nDetecção Pronta (Foco em Recall)")
while True:
    msg = input("\nDigite a mensagem (ou 'sair'): ")
    if msg.lower() == 'sair': break
    analisar_mensagem(msg)