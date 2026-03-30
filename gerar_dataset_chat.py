import pandas as pd
from google import genai
import time
import os

# CONFIG API
CHAVE_API = "AIzaSyASjiH_kC0bsG7uUS1d7V1qmhMJgxgkDHQ" 
client = genai.Client(api_key=CHAVE_API)
MODELO_ESCOLHIDO = 'gemini-2.5-flash' 

# ARQUIVOS
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
    print("Verificando progresso do dataset...")
    df_orig = pd.read_csv(ARQUIVO_ORIGINAL)
    
    # Limpeza padrão para garantir que a ordem seja sempre a mesma
    df_orig = df_orig.drop_duplicates(subset=['texto_relato']).dropna(subset=['texto_relato'])
    df_orig = df_orig[df_orig['texto_relato'].str.len() > 15].reset_index(drop=True)
    
    # Lógica de Memória Mestre: Conta quantas linhas já existem no arquivo final
    linhas_feitas = 0
    if os.path.exists(ARQUIVO_SINTETICO):
        df_saida = pd.read_csv(ARQUIVO_SINTETICO)
        linhas_feitas = len(df_saida)
        print(f"Memória: O arquivo já possui {linhas_feitas} mensagens prontas.")

    if linhas_feitas >= len(df_orig):
        print("Todos os relatos do arquivo original já foram processados!")
        return

    # PULA as linhas que já foram feitas e pega os PRÓXIMOS 15
    df_pendente = df_orig.iloc[linhas_feitas:]
    lote_atual = df_pendente.head(15)
    
    print(f"Iniciando processamento de {len(lote_atual)} itens (Linha {linhas_feitas + 1} em diante)...")

    for _, row in lote_atual.iterrows():
        print(f"Traduzindo: {row['texto_relato'][:50]}...")
        txt_gerado = sintetizar_mensagem_rapida(row['texto_relato'], row['label_ia'])
        
        if txt_gerado:
            # SALVAMENTO IMEDIATO LINHA A LINHA
            nova_linha = pd.DataFrame([{
                'numero': row['numero'],
                'mensagem_chat_sintetica': txt_gerado,
                'label_ia': row['label_ia']
            }])
            
            # Checa se o arquivo já existe para saber se bota o cabeçalho ou não
            precisa_cabecalho = not os.path.exists(ARQUIVO_SINTETICO)
            
            # Salva no arquivo instantaneamente
            nova_linha.to_csv(ARQUIVO_SINTETICO, mode='a', index=False, header=precisa_cabecalho, encoding='utf-8-sig')
            print("  ↳ Mensagem salva no CSV com sucesso!")
            
        time.sleep(4.5)

    print(f"\nLote finalizado! Pode rodar de novo para pegar os próximos.")

if __name__ == "__main__":
    processar_lote_15()