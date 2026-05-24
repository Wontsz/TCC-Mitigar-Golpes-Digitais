import pandas as pd
from google import genai
import time
import os

# CONFIG API
CHAVE_API = "AIzaSyCHUUd1KUyh3ATCdAZgXl8MgWus0-Y30uY" 
client = genai.Client(api_key=CHAVE_API)
MODELO_ESCOLHIDO = 'gemini-2.5-flash'

# ARQUIVOS
DIRETORIO_TCC = r"C:\Users\mateu\Documents\GitHub\TCC-Mitigar-Golpes-Digitais"
ARQUIVO_ORIGINAL = os.path.join(DIRETORIO_TCC, 'dataset_vincular_numero_relato.csv')
ARQUIVO_SINTETICO = os.path.join(DIRETORIO_TCC, 'dataset_mensagens_chat.csv')

def sintetizar_mensagem_rapida(relato, label):
    prompt = f"""
                “Você é um especialista em cibersegurança brasileira. Sua tarefa é gerar UMA mensagem curta de SMS ou WhatsApp simulando uma tentativa real de smishing (golpe por mensagem), com base no relato de uma vítima.
            CONTEXTO DA TAREFA:
            A mensagem será usada para treinar um modelo de IA que detecta golpes. Para que o modelo aprenda a SEMÂNTICA do golpe (e não apenas o estilo), as mensagens fraudulentas devem ser realistas e variadas em tom, evitando exageros estilísticos que não correspondem ao que criminosos reais enviam.
            REGRAS OBRIGATÓRIAS:
            1. ESTILO DIVERSIFICADO — varie o tom da mensagem entre as gerações:
                • Algumas em tom URGENTE e dramático (com caixa alta e exclamações)
                • Algumas em tom CALMO e profissional (sem caixa alta, redação formal)
                • Algumas em tom INFORMAL e amigável (como um conhecido)
                • Algumas em tom BUROCRÁTICO (linguagem jurídica ou institucional)
            Cerca de 40% do total deve usar tom calmo/profissional, justamente porque golpes sofisticados imitam comunicações reais.
            2. CONTEÚDO CARACTERÍSTICO DE GOLPE — a mensagem deve conter pelo menos um destes elementos, que são o que de fato caracteriza smishing:
                • Pedido de clique em link suspeito (encurtador, domínio estranho)
                • Pedido de dados sensíveis (senha, código, dados bancários)
                • Pedido de transferência via Pix para número desconhecido
                • Falsa identidade (se passar por banco, órgão público, parente, empresa)
                • Promessa de prêmio ou benefício mediante ação rápida
                • Ameaça de bloqueio, processo ou multa para induzir reação
            3. VARIE OS CENÁRIOS — use diferentes tipos de golpe a cada geração:
                • Falso parente: "Oi, troquei de número, preciso de uma ajuda urgente"
                • Falsa entrega: encomenda retida, taxa pendente
                • Falso emprego: vaga selecionada, cadastro via link
                • Milhas e pontos: pontos a vencer, resgate por link
                • Bancário: compra não reconhecida, bloqueio de conta
                • Crédito e apostas: valor creditado, saque via link
                • Órgão público: multa, intimação, restrição de CPF
                • Cobrança fraudulenta: dívida inexistente com urgência
            4. NÃO INCLUA — nunca use as palavras "spam", "golpe", "fraude" ou "denúncia" no texto. Aja estritamente como o remetente real (criminoso ou empresa falsa).
            5. IGNORE rótulos no início do relato (ex.: "tentativa de golpe", "fraude detectada"). Use apenas a narrativa em si.
            6. FORMATO DA RESPOSTA — responda APENAS com o texto final da mensagem, como apareceria no celular da vítima. Sem aspas, sem rótulos, sem explicação.”

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
    
    # --- MEMÓRIA INTELIGENTE BLINDADA (IGNORANDO DADOS INJETADOS) ---
    linhas_feitas = 0
    precisa_cabecalho = True
    
    if os.path.exists(ARQUIVO_SINTETICO):
        with open(ARQUIVO_SINTETICO, 'r', encoding='utf-8-sig') as f:
            linhas = f.readlines()
            if len(linhas) > 0:
                precisa_cabecalho = False
                # Pula o cabeçalho e verifica linha por linha
                for linha in linhas[1:]:
                    linha_limpa = linha.strip()
                    if linha_limpa:
                        # Isola a primeira coluna e remove aspas e espaços
                        primeira_coluna = linha_limpa.split(',')[0].replace('"', '').strip()
                        
                        # Só conta se o número for diferente de 000
                        if primeira_coluna != '000':
                            linhas_feitas += 1
                            
        print(f"Memória: {linhas_feitas} mensagens extraídas do Tellows processadas (ignorando as injetadas).")

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