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
    Você é um especialista em cibersegurança e engenharia social. 
    TRANSFORME o relato abaixo em uma MENSAGEM CURTA de SMS ou WhatsApp.

    Regras CRÍTICAS:
    1. IGNORE TOTALMENTE rótulos de classificação no início do relato (ex: "de golpe", "Tentativa de golpe", "spam"). Leia apenas a historinha.
    2. NUNCA inclua palavras como "spam", "golpe" ou "denúncia" no texto gerado. Aja como o criminoso ou a empresa.
    3. Se Label = 1 (Golpe): DIVERSIFIQUE AO MÁXIMO! Baseie-se no relato, mas varie os estilos de ataque. Use formatos como:
       - Falso Parente: "Oi mãe/pai, troquei de numero, preciso de dinheiro!"
       - Falsa Entrega/Correios: "Sua encomenda foi retida na alfândega. Pague a taxa de R$ XXX,XX em: [link]"
       - Falso Emprego: "Você foi selecionado para trabalhar meio período ganhando R$ 500/dia. Acesse: [link]"
       - Milhas/Pontos: "Seus 15.000 pontos Livelo vencem HOJE. Resgate por Pix em: [link]"
       - Bancário clássico: Falsa compra, falso Pix agendado ou bloqueio de conta.
       - Crédito falso: "R$XXX creditados, jogue ou saque, acesse [link]."
       - Sites de apostas: "Voce recebeu 50 rodadas no [nome do jogo]! Credito de R$XXX ja adicionados [link]."
    4. Se Label = 0 (Seguro): Crie um SMS de telemarketing padrão, cobrança legítima educada ou informativo de operadora, sem ameaças.
    5. Responda APENAS com o texto final da mensagem nua e crua.

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

def processar_lote_20():
    print("Verificando progresso do dataset...")
    df_orig = pd.read_csv(ARQUIVO_ORIGINAL)
    
    # Limpeza padrão
    df_orig = df_orig.drop_duplicates(subset=['texto_relato']).dropna(subset=['texto_relato'])
    df_orig = df_orig[df_orig['texto_relato'].str.len() > 20].reset_index(drop=True)
    
    # --- MEMÓRIA NATIVA DO PYTHON (LENDO AS LINHAS DO ARQUIVO) ---
    linhas_feitas = 0
    precisa_cabecalho = True
    
    if os.path.exists(ARQUIVO_SINTETICO):
        with open(ARQUIVO_SINTETICO, 'r', encoding='utf-8-sig') as f:
            linhas = f.readlines()
            if len(linhas) > 0:
                precisa_cabecalho = False
                linhas_feitas = max(0, len(linhas) - 1) # Subtrai o cabeçalho
        print(f"Memória: O arquivo já possui {linhas_feitas} mensagens processadas.")

    if linhas_feitas >= len(df_orig):
        print("Todos os relatos já foram processados!")
        return

    # Pula as linhas já feitas
    df_pendente = df_orig.iloc[linhas_feitas:]
    lote_atual = df_pendente.head(20)
    
    print(f"Iniciando processamento de {len(lote_atual)} itens (Da linha {linhas_feitas + 1} em diante)...")

    # Abre o arquivo UMA VEZ e vai injetando o texto cru, garantindo a formatação
    with open(ARQUIVO_SINTETICO, 'a', encoding='utf-8-sig') as arquivo_saida:
        if precisa_cabecalho:
            arquivo_saida.write("numero,mensagem_chat_sintetica,label_ia\n")
            
        for _, row in lote_atual.iterrows():
            print(f"Traduzindo: {row['texto_relato'][:50]}...")
            txt_gerado = sintetizar_mensagem_rapida(row['texto_relato'], row['label_ia'])
            
            if txt_gerado:
                # Remove aspas duplas de dentro do texto e quebras de linha para não quebrar o CSV
                txt_gerado = txt_gerado.replace('"', "'").replace('\n', ' ')
                
                # A MÁGICA AQUI: O texto é forçado a ter aspas, os números não.
                linha_formatada = f'{row["numero"]},"{txt_gerado}",{row["label_ia"]}\n'
                
                arquivo_saida.write(linha_formatada)
                arquivo_saida.flush() # Salva fisicamente no HD na mesma hora
                print("  ↳ Salvo no CSV!")
                
            time.sleep(4.5)

    print(f"\nLote finalizado! Pode rodar de novo para pegar os próximos.")

if __name__ == "__main__":
    processar_lote_20()