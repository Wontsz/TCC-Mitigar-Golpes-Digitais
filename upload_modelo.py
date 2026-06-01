from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Caminho para o model.safetensors
pasta_local = "C:/Users/mateu/Documents/GitHub/TCC-Mitigar-Golpes-Digitais/modelo_smishing_final"

print("A carregar o modelo local na memória...")

modelo = AutoModelForSequenceClassification.from_pretrained(pasta_local)
tokenizer = AutoTokenizer.from_pretrained(pasta_local)

nome_do_repositorio_nuvem = "Wontsimt/detector-phishing-v1"

print(f"A iniciar o upload para: {nome_do_repositorio_nuvem}")
print("Enviando para a nuvem do Hugging Face...")

modelo.push_to_hub(nome_do_repositorio_nuvem)
tokenizer.push_to_hub(nome_do_repositorio_nuvem)

print("Upload concluído")