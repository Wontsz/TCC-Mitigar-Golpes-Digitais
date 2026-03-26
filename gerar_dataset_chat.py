import pandas as pd
from google import genai
import time
import os

# --- 1. CONFIGURAÇÃO DA API ---
CHAVE_API = "AIzaSyASjiH_kC0bsG7uUS1d7V1qmhMJgxgkDHQ" 
client = genai.Client(api_key=CHAVE_API)
MODELO_ESCOLHIDO = 'gemini-2.5-flash'

# --- 2. CONFIGURAÇÃO DE ARQUIVOS ---
DIRETORIO_TCC = r"C:\Users\mateu\Documents\GitHub\TCC-Mitigar-Golpes-Digitais"
ARQUIVO_ORIGINAL = os.path.join(DIRETORIO_TCC, 'dataset_vincular_numero_relato.csv')
ARQUIVO_SINTETICO = os.path.join(DIRETORIO_TCC, 'dataset_mensagens_chat.csv')

def sintetizar_mensagem_chat(relato, label):
    prompt = f"""
    Você é um especialista em cibersegurança construindo um dataset para treinar uma IA de detecção de fraudes.
    Abaixo está um relato de um usuário que recebeu uma ligação ou SMS.
    Sua tarefa é TRANSFORMAR essa reclamação na MENSAGEM EXATA de WhatsApp ou SMS que a pessoa recebeu.

    Regras OBRIGATÓRIAS:
    1. Se Label = 1 (Golpe/Ameaça): Escreva como o golpista. Use engenharia social, crie senso de urgência, medo ou falsa oportunidade. Finja ser o banco, a polícia ou a empresa citada. Inclua links falsos (ex: bit.ly/123) ou 0800 se fizer sentido.
    2. Se Label = 0 (Telemarketing/Legítimo/Engano): Crie uma mensagem de texto (SMS/WhatsApp) de uma operadora de celular oferecendo um plano chato, ou uma mensagem de cobrança legítima e educada, sem ameaças. NÃO use "Alô?", crie um texto lido na tela.
    3. Responda APENAS com o texto da mensagem simulada. Não use aspas, não dê explicações, não escreva "Mensagem:". Quero apenas o texto cru que apareceria na tela do celular.

    Label do caso: {label}
    Relato da Vítima: "{relato}"
    """
    
    # Tentativa com blindagem contra limite de cota (Erro 429)
    while True:
        try:
            resposta = client.models.generate_content(
                model=MODELO_ESCOLHIDO,
                contents=prompt
            )
            return resposta.text.strip()
        except Exception as e:
            erro = str(e)
            if '429' in erro or 'quota' in erro.lower():
                print("  [!] Limite da API atingido. Pausando por 60 segundos para esfriar o servidor...")
                time.sleep(60) # Espera 1 minuto e tenta de novo o mesmo relato
            else:
                print(f"  [x] Erro desconhecido: {e}")
                return None

def criar_dataset_definitivo():
    print("Iniciando a Fábrica de Engenharia Social com Blindagem...\n")
    df = pd.read_csv(ARQUIVO_ORIGINAL)
    
    # Limpeza básica
    df = df.drop_duplicates(subset=['texto_relato']).dropna(subset=['texto_relato'])
    df = df[df['texto_relato'].str.len() > 15].reset_index(drop=True)
    
    total = len(df)
    print(f"Total de relatos para traduzir: {total}")
    
    # Criamos a coluna ou lemos o que já foi salvo antes
    if 'mensagem_chat_sintetica' not in df.columns:
        df['mensagem_chat_sintetica'] = ""
    
    for index, row in df.head(15).iterrows():
        # Se já traduziu essa linha antes, pula (bom para caso você feche o script no meio)
        if pd.notna(row['mensagem_chat_sintetica']) and str(row['mensagem_chat_sintetica']).strip() != "":
            continue

        print(f"\n[{index+1}/{total}] Traduzindo: {row['texto_relato'][:50]}...")
        
        mensagem_gerada = sintetizar_mensagem_chat(row['texto_relato'], row['label_ia'])
        
        if mensagem_gerada:
            df.at[index, 'mensagem_chat_sintetica'] = mensagem_gerada
            print(f"↳ MENSAGEM GERADA: {mensagem_gerada}")
            
            # Salva o arquivo a cada linha gerada. Se cair a energia, você não perde nada!
            df_salvar = df[df['mensagem_chat_sintetica'] != ""] 
            df_salvar = df_salvar[['numero', 'mensagem_chat_sintetica', 'label_ia']]
            df_salvar.to_csv(ARQUIVO_SINTETICO, index=False, encoding='utf-8-sig')
            
        time.sleep(5) # Aumentamos para 5 segundos para respeitar o limite de 15 RPM do Google

    print(f"\n✅ Fábrica concluída! Todo o dataset foi gerado e salvo em: {ARQUIVO_SINTETICO}")

if __name__ == "__main__":
    criar_dataset_definitivo()