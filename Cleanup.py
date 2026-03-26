import pandas as pd

# Caminho do seu arquivo (ajuste se necessário)
ARQUIVO_MESTRE = r"C:\Users\mateu\Documents\GitHub\TCC-Mitigar-Golpes-Digitais\dataset_vincular_numero_relato.csv"

def remover_textos_repetidos():
    print("Carregando o dataset...")
    df = pd.read_csv(ARQUIVO_MESTRE)
    
    total_antes = len(df)
    print(f"Total de relatos antes da limpeza: {total_antes}")
    
    # O segredo: Apaga as linhas onde o 'texto_relato' for exatamente igual a outro que já passou
    # O 'keep="first"' garante que ele salva o primeiro que encontrar e joga o resto fora.
    df_limpo = df.drop_duplicates(subset=['texto_relato'], keep='first')
    
    total_depois = len(df_limpo)
    removidos = total_antes - total_depois
    
    print(f"🧹 Limpeza concluída! Foram removidos {removidos} relatos genéricos/repetidos.")
    print(f"✅ Novo total de relatos únicos de verdade: {total_depois}")
    
    # Salva por cima do arquivo original
    df_limpo.to_csv(ARQUIVO_MESTRE, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    remover_textos_repetidos()