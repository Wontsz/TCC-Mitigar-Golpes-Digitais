from transformers import AutoTokenizer, AutoModel
import torch

print("--- Iniciando teste do BERTimbau ---")

try:
    # 1. Definir o modelo (BERTimbau Base)
    model_name = "neuralmind/bert-base-portuguese-cased"

    # 2. Tentar carregar o Tokenizador e o Modelo
    print("Carregando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    texto = "Isso é um teste para o meu TCC de golpes digitais."
    inputs = tokenizer(texto, return_tensors="pt")
    
    print("\n SUCESSO!")
    print(f"O modelo foi carregado e identificou {len(inputs['input_ids'][0])} tokens na frase.")

except Exception as e:
    print(f"\n ERRO")
    print(f"Detalhe do erro: {e}")