import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re

# --- CONFIGURAÇÃO ---
DIRETORIO_TCC = r"C:\Users\mateu\Documents\GitHub\TCC-Mitigar-Golpes-Digitais"
ARQUIVO_MESTRE = os.path.join(DIRETORIO_TCC, 'dataset_vincular_numero_relato.csv')
ARQUIVO_BLACKLIST = os.path.join(DIRETORIO_TCC, 'blacklist_numeros.csv')

# Lista de Gatilhos
GATILHOS = [
    'golpe', 'pix', 'falso', '0800', 'bloqueio', 'clonar', 'urgente', 
    'banco', 'penhora', 'prisão', 'suspensão', 'pendência', 'fatura', 
    'facção', 'ameaça', 'polícia', 'detran', 'alfândega', 'correios', 
    'taxa', 'verificar', 'confirmar compra', 'indevida', 'indevido',
    'cartao', 'dados', 'spam', 'urgência', 'apreensão', 'carteira',
    'advogado', 'justiça', 'bradesco', 'itau', 'nubank', 'bb', 
    'banco do brasil', 'tim', 'claro', 'vivo', 'fraude',
    'liga e desliga', 'robo', 'fraudulento', 'fraudulenta'
]

def lapidar_texto_ia(texto):
    texto = str(texto)
    texto = re.sub(r'^[A-Z][a-z]+[A-Z][a-z]+', '', texto)
    padroes = [
        r'.*?identificou o número.*?\d+', r'.*?avaliou.*?como', 
        r'.*?foi nomeado como', r'.*?desconhecido', r'deAndroid App', 
        r'deiPhone App', r'https?://\S+', r'‎\+?\d+', 
        r'\d{2}/\d{2}/\d{2,4}', r'\d{2}:\d{2}', r'responder', r'respondeu'
    ]
    for p in padroes:
        texto = re.sub(p, '', texto, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', texto).strip()

def contar_gatilhos(texto):
    texto_low = texto.lower()
    return sum(1 for g in GATILHOS if g in texto_low)

def minerar_vincular_dados():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    base_url = "https://www.tellows.com.br"
    
    try:
        if not os.path.exists(DIRETORIO_TCC): os.makedirs(DIRETORIO_TCC)

        print("\n" + "="*50)
        print("WEB SCRAPING INICIADO!")
        print("="*50)
        
        res = requests.get(f"{base_url}/stats", headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = list(set([a['href'] for a in soup.find_all('a', href=True) if '/num/' in a['href']]))[:15]

        novos_relatos = []

        for link in links:
            num_limpo = link.split('/')[-1].replace('%2B', '').replace('+', '')
            print(f"Analisando: {num_limpo}...")
            time.sleep(2)
            
            try:
                detalhe = requests.get(base_url + link, headers=headers, timeout=10)
                s_det = BeautifulSoup(detalhe.text, 'html.parser')

                # --- TÉCNICA SNIPER: ANCORAGEM SEMÂNTICA ---
                score_site = 0
                ancora = s_det.find(string=re.compile(r'Classifica..o para|Avalia..o:', re.IGNORECASE))
                
                if ancora:
                    caixa_principal = ancora.find_parent('div')
                    if caixa_principal:
                        img_escudo = caixa_principal.find('img', src=re.compile(r'/s[1-9]\.(jpg|png)', re.I))
                        
                        if not img_escudo:
                            caixa_principal_nivel2 = caixa_principal.find_parent('div')
                            if caixa_principal_nivel2:
                                img_escudo = caixa_principal_nivel2.find('img', src=re.compile(r'/s[1-9]\.(jpg|png)', re.I))

                        if img_escudo:
                            src = img_escudo.get('src', '')
                            match = re.search(r's([1-9])\.(jpg|png)', src, re.IGNORECASE)
                            if match:
                                score_site = int(match.group(1))

                # --- FALLBACK ---
                if score_site == 0:
                    todas_imagens = s_det.find_all('img', src=re.compile(r'/s[1-9]\.(jpg|png)', re.I))
                    if todas_imagens:
                        match = re.search(r's([1-9])\.(jpg|png)', todas_imagens[0].get('src', ''), re.IGNORECASE)
                        if match:
                            score_site = int(match.group(1))

                # Varredura de Comentários e IA
                comentarios = s_det.find_all(['div', 'p'], class_=['comment-body', 'comment-content', 'comment-text'])
                for comment in comentarios:
                    texto_original = comment.get_text(strip=True)
                    if len(texto_original) > 30:
                        txt = lapidar_texto_ia(texto_original)
                        if len(txt) > 15:
                            gatilhos_encontrados = contar_gatilhos(txt)
                            label = 1 if (score_site >= 6 or gatilhos_encontrados > 0) else 0
                            
                            novos_relatos.append({
                                'numero': num_limpo,
                                'score_site': score_site,
                                'texto_relato': txt,
                                'qtd_gatilhos': gatilhos_encontrados,
                                'label_ia': label,
                                'data_coleta': time.strftime('%d/%m/%Y')
                            })
            except: continue

        if novos_relatos:
            df_novos = pd.DataFrame(novos_relatos)
            if os.path.exists(ARQUIVO_MESTRE):
                df_antigo = pd.read_csv(ARQUIVO_MESTRE)
                df_final_mestre = pd.concat([df_antigo, df_novos]).drop_duplicates(subset=['numero', 'texto_relato'])
            else:
                df_final_mestre = df_novos
            
            # Limpeza preventiva extra
            df_final_mestre = df_final_mestre[df_final_mestre['score_site'] <= 9]
            df_final_mestre.to_csv(ARQUIVO_MESTRE, index=False, encoding='utf-8-sig')

            # --- ATUALIZAÇÃO DA BLACKLIST ---
            blacklist = df_final_mestre.groupby('numero').agg({
                'score_site': 'last',
                'qtd_gatilhos': 'sum', 
                'label_ia': 'max'      
            }).reset_index()

            blacklist['score_final_tcc'] = blacklist['score_site'] + (blacklist['qtd_gatilhos'] // 2)
            blacklist.loc[blacklist['score_final_tcc'] > 10, 'score_final_tcc'] = 10
            
            blacklist = blacklist.sort_values(by='score_final_tcc', ascending=False)
            blacklist.to_csv(ARQUIVO_BLACKLIST, index=False, encoding='utf-8-sig')

            # --- PAINEL DE RESUMO DA EXECUÇÃO ---
            print("\n" + "="*50)
            print("***SUCESSO! MINERAÇÃO CONCLUÍDA***")
            print("="*50)
            print(f"RESUMO DA EXECUÇÃO:")
            print(f"Números únicos analisados hoje: {len(links)}")
            print(f"Novos relatos minerados agora: {len(novos_relatos)}")
            print(f"Total de relatos no Dataset Mestre: {len(df_final_mestre)}")
            print(f"Total de números na Blacklist: {len(blacklist)}")
            print("="*50)
        else:
            print("\n! Nenhum relato novo encontrado nesta execução.!")

    except Exception as e: print(f"Erro: {e}")

if __name__ == "__main__":
    minerar_vincular_dados()