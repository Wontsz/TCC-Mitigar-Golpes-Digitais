import pandas as pd
import re
import os

# Caminho do seu arquivo
ARQUIVO_MESTRE = r"C:\Users\mateu\Documents\Faculdade\TCC\dataset_vincular_numero_relato.csv"
GATILHOS = [
    'golpe', 'pix', 'falso', '0800', 'bloqueio', 'clonar', 'urgente', 
    'banco', 'penhora', 'prisão', 'suspensão', 'pendência', 'fatura', 
    'facção', 'ameaça', 'polícia', 'detran', 'alfândega', 'correios', 
    'taxa', 'verificar', 'confirmar compra', 'indevida', 'cartao', 'dados',
    'spam', 'urgência', 'apreensão', 'carteira', 'advogado', 'justiça'
]

def limpar_profundo(texto):
    texto = str(texto)
    # Remove restos de tags e caracteres aleatórios
    texto = re.sub(r'[@#\$%\^&\*\(\)\[\]\{\}]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def recalcular_gatilhos(texto):
    texto_low = str(texto).lower()
    return sum(1 for g in GATILHOS if g in texto_low)

if os.path.exists(ARQUIVO_MESTRE):
    df = pd.read_csv(ARQUIVO_MESTRE)
    
    # 1. Limpeza de texto
    df['texto_relato'] = df['texto_relato'].apply(limpar_profundo)
    
    # 2. Recalcula gatilhos para preencher os que estavam vazios
    df['qtd_gatilhos'] = df['texto_relato'].apply(recalcular_gatilhos)
    
    # 3. Corrige o Label (Se tem gatilho ou score alto, é 1)
    df['label_ia'] = df.apply(lambda row: 1 if (row['score_site'] >= 6 or row['qtd_gatilhos'] > 0) else 0, axis=1)
    
    # 4. Remove linhas inúteis (muito curtas ou sem letras)
    df = df[df['texto_relato'].str.len() > 15]
    
    # 5. Remove duplicatas
    df = df.drop_duplicates(subset=['numero', 'texto_relato'])
    
    df.to_csv(ARQUIVO_MESTRE, index=False, encoding='utf-8-sig')
    print(f"Faxina concluída! {len(df)} linhas prontas para o Google Colab.")
else:
    print("Arquivo não encontrado para faxina.")