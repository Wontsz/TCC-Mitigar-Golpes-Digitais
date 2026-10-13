"""Caminhos e divisão de dados compartilhados pelos scripts de conversão/avaliação.

Todos os caminhos são relativos ao repositório, então os scripts podem ser
rodados de qualquer pasta (ex.: `python conversao_onnx/exportar_onnx.py`).
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

PASTA = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PASTA)

CSV = os.path.join(REPO, "datasets", "dataset_v7.csv")
MODELO_HF_ID = "Wontsimt/detector-phishing-v3-final"
MODELO_HF_DIR = os.path.join(PASTA, "modelo_hf")      # baixado por baixar_modelo.py (fora do git)
SAIDA = os.path.join(PASTA, "saida")                  # .onnx gerados (fora do git)
FP32 = os.path.join(SAIDA, "model_fp32.onnx")
INT8 = os.path.join(SAIDA, "model.onnx")              # INT8 dinâmico padrão
VARIANTES = os.path.join(SAIDA, "variantes")
V1_PER_CHANNEL = os.path.join(VARIANTES, "v1_per_channel.onnx")  # modelo usado no app

ENTRADAS = ("input_ids", "attention_mask", "token_type_ids")
MAX_LENGTH = 128


def carregar_divisao():
    """Mesma divisão do fine_tuning.py: 70/15/15, estratificada, random_state=42.

    Retorna (df_train, df_val, df_test) com as colunas renomeadas para text/label.
    """
    df = pd.read_csv(CSV).rename(columns={"mensagem_chat_sintetica": "text", "label_ia": "label"})
    df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42)
    return df_train, df_val, df_test
