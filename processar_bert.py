import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Configuração do Modelo (BERTimbau)
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def gerar_embedding(texto):
    # Prepara o texto para o BERT
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].numpy()

try:
    df = pd.read_csv('dataset_mensagens_chat.csv')
    coluna_mensagem = 'text' # ou 'mensagem', 'content', etc.
    
    print(f"Processando {len(df)} mensagens com BERTimbau...")
    
    exemplo = df[coluna_mensagem].iloc[0]
    vetor = gerar_embedding(exemplo)
    
    print("\n[SUCESSO] BERT carregado e funcional!")
    print(f"Exemplo de mensagem: {exemplo}")
    print(f"Dimensão do vetor gerado: {vetor.shape} (768 dimensões)")

except Exception as e:
    print(f"Erro ao carregar o projeto: {e}")