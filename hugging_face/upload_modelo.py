import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

pasta_local = "./modelo_smishing_final"
repo = "victorcpregno/detector-smishing-v7"

print("Carregando o modelo local...")
modelo = AutoModelForSequenceClassification.from_pretrained(pasta_local)
tokenizer = AutoTokenizer.from_pretrained(pasta_local)

print(f"Enviando para: {repo}")
modelo.push_to_hub(repo, token=os.environ["HF_TOKEN"])
tokenizer.push_to_hub(repo, token=os.environ["HF_TOKEN"])
print("Upload concluido")