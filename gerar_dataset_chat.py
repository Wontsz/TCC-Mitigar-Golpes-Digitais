import pandas as pd
from google import genai
import time
import os

# --- 1. CONFIGURAÇÃO DA API ---
CHAVE_API = "AIzaSyASjiH_kC0bsG7uUS1d7V1qmhMJgxgkDHQ" 
client = genai.Client(api_key=CHAVE_API)
# Utilize o modelo que estiver com cota liberada na sua conta
MODELO_ESCOLHIDO = 'gemini-2.0-flash' 

# --- 2. CONFIGURAÇÃO DE ARQUIVOS ---
DIRETORIO_TCC = r"C:\Users\mateu\Documents\GitHub\TCC-Mitigar-Golpes-Digitais"
ARQUIVO_ORIGINAL = os.path.join(DIRETORIO_TCC, 'dataset_vincular_numero_relato.csv')
ARQUIVO_SINTETICO = os.path.join(DIRETORIO_TCC, 'dataset_mensagens_chat.csv')

def sintetizar_mensagem_rapida(relato, label):
    prompt = f"""
    Você é um especialista em cibersegurança. 
    TRANSFORME o relato abaixo em uma MENSAGEM CURTA de SMS ou Alerta de WhatsApp.

    Regras:
    1. Se Label = 1 (Golpe): Crie um texto de impacto direto (Smishing). Use ganchos de: bloqueio de conta, compra suspeita, processo judicial ou promoção falsa. Inclua links encurtados ou números 0800.
    2. Se Label = 0 (Seguro): Crie um SMS de telemarketing padrão ou informativo de operadora, sem tom de urgência ou ameaça.
    3. Responda APENAS com o texto final da mensagem.

    Label: {label}
    Relato: "{relato}"
    """
    
    while True:
        try:
            resposta = client.models.generate_content(
                model=MODELO_ESCOLHIDO,
                contents=prompt
            )
            return resposta.text.strip()
        except Exception as e:
            if '429' in str(e):
                print("  [!] Cota atingida. Pausando 60s...")
                time.sleep(60)
            else:
                print(f"  [x] Erro: {e}")
                return None

def processar_lote_15():
    print("🚀 Verificando progresso do dataset...")
    df_orig = pd.read_csv(ARQUIVO_ORIGINAL)
    
    # Limpeza de duplicatas no arquivo de origem
    df_orig = df_orig.drop_duplicates(subset=['texto_relato']).dropna(subset=['texto_relato'])
    
    # Lógica de Memória: Verifica o que já está no arquivo de saída
    relatos_processados = []
    if os.path.exists(ARQUIVO_SINTETICO):
        df_saida = pd.read_csv(ARQUIVO_SINTETICO)
        if 'texto_relato' in df_saida.columns:
            relatos_processados = df_saida['texto_relato'].tolist()
            print(f"📊 {len(relatos_processados)} itens já foram concluídos anteriormente.")

    # Filtra apenas o que ainda não foi traduzido
    df_pendente = df_orig[~df_orig['texto_relato'].isin(relatos_processados)].copy()
    
    if df_pendente.empty:
        print("✅ Todos os relatos do arquivo original já foram processados!")
        return

    # Seleciona os PRÓXIMOS 15 da fila
    lote_atual = df_pendente.head(15)
    print(f"📦 Iniciando processamento dos próximos {len(lote_atual)} itens...")

    resultados = []
    for _, row in lote_atual.iterrows():
        print(f"Traduzindo: {row['texto_relato'][:50]}...")
        txt_gerado = sintetizar_mensagem_rapida(row['texto_relato'], row['label_ia'])
        
        if txt_gerado:
            resultados.append({
                'numero': row['numero'],
                'texto_relato': row['texto_relato'], # Essencial para o sistema de memória
                'mensagem_chat_sintetica': txt_gerado,
                'label_ia': row['label_ia']
            })
        time.sleep(4.5) # Respeitando o limite de 15 requisições por minuto

    # Salva no modo 'append' (adiciona ao final do arquivo)
    if resultados:
        new_df = pd.DataFrame(resultados)
        header_needed = not os.path.exists(ARQUIVO_SINTETICO)
        new_df.to_csv(ARQUIVO_SINTETICO, mode='a', index=False, header=header_needed, encoding='utf-8-sig')
        print(f"\n✨ Lote finalizado e salvo em: {ARQUIVO_SINTETICO}")

if __name__ == "__main__":
    processar_lote_15()