"""Métricas nos limiares 0,5 / 0,4 / 0,3 no conjunto de teste (mesma divisão do fine_tuning.py).

Avalia o modelo em PyTorch e, se existirem, o ONNX INT8 padrão e a v1 per-channel.
Modelos ONNX extras podem ser passados como argumentos NOME=CAMINHO, ex.:
    python conversao_onnx/avaliar_modelos.py "app antigo=C:/caminho/model_2026-09-09.onnx"

Inferência em lotes de 32 (para o PyTorch o lote não altera o resultado; para o
INT8 dinâmico altera um pouco: ver comparar_variantes.py, que avalia 1 mensagem por vez).
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import scipy.special
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from comum import ENTRADAS, INT8, MAX_LENGTH, MODELO_HF_DIR, V1_PER_CHANNEL, carregar_divisao

LIMIARES = [0.5, 0.4, 0.3]
LOTE = 32

df_train, df_val, df_test = carregar_divisao()
textos = df_test["text"].tolist()
y = df_test["label"].to_numpy()
print(f"teste: {len(y)} mensagens (golpe={int(y.sum())}, legítima={int((y == 0).sum())}) | "
      f"treino {len(df_train)} | validação {len(df_val)}")

tok = AutoTokenizer.from_pretrained(MODELO_HF_DIR)
enc = tok(textos, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="np")
enc = {k: enc[k].astype(np.int64) for k in ENTRADAS}


def prob_pytorch():
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_HF_DIR).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(textos), LOTE):
            out.append(model(**{k: torch.from_numpy(enc[k][i:i + LOTE]) for k in ENTRADAS}).logits.numpy())
    return scipy.special.softmax(np.concatenate(out), axis=1)[:, 1]


def prob_onnx(caminho):
    sess = ort.InferenceSession(caminho, providers=["CPUExecutionProvider"])
    nomes = [x.name for x in sess.get_inputs()]
    out = [sess.run(None, {k: enc[k][i:i + LOTE] for k in nomes})[0] for i in range(0, len(textos), LOTE)]
    return scipy.special.softmax(np.concatenate(out), axis=1)[:, 1]


def tabela(nome, p):
    print(f"\n=== {nome} ===")
    print(f"{'limiar':>6} | {'acurácia':>8} | {'precisão':>8} | {'recall':>8} | {'F1':>8} | [[VN FP] [FN VP]]")
    for t in LIMIARES:
        pred = (p >= t).astype(int)
        print(f"{t:>6} | {accuracy_score(y, pred):8.4f} | {precision_score(y, pred, zero_division=0):8.4f} | "
              f"{recall_score(y, pred, zero_division=0):8.4f} | {f1_score(y, pred, zero_division=0):8.4f} | "
              f"{confusion_matrix(y, pred).tolist()}")


tabela("PyTorch FP32", prob_pytorch())
for nome, caminho in [("ONNX INT8 padrão", INT8), ("ONNX v1 per-channel", V1_PER_CHANNEL)]:
    if os.path.exists(caminho):
        tabela(nome, prob_onnx(caminho))
for arg in sys.argv[1:]:
    nome, caminho = arg.split("=", 1)
    tabela(nome, prob_onnx(caminho))
