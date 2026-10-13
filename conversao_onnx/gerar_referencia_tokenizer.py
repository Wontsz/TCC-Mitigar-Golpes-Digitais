"""Gera a referência de tokenização do HF usada no teste unitário do app (WordPieceTokenizerTest).

Uso:
    python conversao_onnx/gerar_referencia_tokenizer.py <saida.tsv>
No app, a saída vai para app/src/test/resources/tokenizer_reference.tsv.

Formato (uma mensagem por linha, separada por TAB):
    nome <TAB> code points (decimal, separados por espaço) <TAB> input_ids <TAB> attention_mask
"""
import sys

import pandas as pd
from transformers import AutoTokenizer

from comum import CSV, MAX_LENGTH, MODELO_HF_DIR

SAIDA = sys.argv[1]

EXTRAS = {
    "reticencias": "URGENTE… sua conta será bloqueada",
    "aspas_curvas_travessao": "Clique “aqui” — já",
    "bullet_underscore": "•CAIXA_INFORMA: bit.ly/x",
    "invisiveis": "Seu Pi⁨x‌ foi blo⁨queado, con‌firme agora",
    "emoji": "Seu pacote chegou 📦 rastreie aqui 👉 bit.ly/x 😀",
    "emoji_zwj_variacao": "Família 👨‍👩‍👧 e coração ❤️",
    "cjk": "Promoção 中文 limitada 𠀀 hoje",
    "nbsp_e_espacos_unicode": "R$ 500,00 agora mesmo　ok",
    "controles": "linha1\r\nlinha2\tcol\u000Bvt\u0085nel\u0000nul�sub",
    "palavra_101_chars": "a" * 101 + " fim",
    "palavra_100_chars": "a" * 100 + " fim",
    "vazia": "",
}

tok = AutoTokenizer.from_pretrained(MODELO_HF_DIR)
df = pd.read_csv(CSV)
textos = [(f"dataset_{i}", str(t)) for i, t in enumerate(df["mensagem_chat_sintetica"])]
textos += list(EXTRAS.items())

with open(SAIDA, "w", encoding="utf-8", newline="\n") as f:
    for nome, texto in textos:
        enc = tok(texto, padding="max_length", truncation=True, max_length=MAX_LENGTH)
        assert all(t == 0 for t in enc["token_type_ids"]), nome
        cps = " ".join(str(ord(c)) for c in texto)
        f.write(f"{nome}\t{cps}\t{' '.join(map(str, enc['input_ids']))}\t{' '.join(map(str, enc['attention_mask']))}\n")

print(f"{len(textos)} mensagens ({len(df)} do dataset + {len(EXTRAS)} extras) -> {SAIDA}")
