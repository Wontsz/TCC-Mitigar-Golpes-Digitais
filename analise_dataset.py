import pandas as pd
import re
from collections import Counter


df = pd.read_csv('dataset_v4_final.csv')


def tokenizar(textos):
    palavras = []
    for t in textos:
        palavras.extend(re.findall(r'\w+', str(t).lower()))
    return Counter(palavras)


c1 = tokenizar(df[df['label_ia'] == 1]['mensagem_chat_sintetica'])
c0 = tokenizar(df[df['label_ia'] == 0]['mensagem_chat_sintetica'])


# Palavras que aparecem muito em uma classe e quase nada na outra
suspeitas = []
for palavra in set(list(c1.keys()) + list(c0.keys())):
    f1, f0 = c1.get(palavra, 0), c0.get(palavra, 0)
    if f1 + f0 < 20:
        continue  # ignora palavras raras
    razao = max(f1, f0) / max(min(f1, f0), 1)
    if razao > 10:
        suspeitas.append((palavra, f1, f0, razao))


suspeitas.sort(key=lambda x: -x[3])
for s in suspeitas[:20]:
    print(s)
