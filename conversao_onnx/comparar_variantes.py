"""Compara PyTorch e variantes ONNX no teste do dataset_v7 (limiar 0,5 por padrão) + tempo por mensagem.

Cada mensagem é inferida SOZINHA (lote = 1), como no app. Isso importa para o
INT8 dinâmico: a escala de quantização das ativações é calculada sobre o lote
inteiro, então avaliar em lotes maiores muda levemente os resultados.

Opções:
    --limiar=0.3     limiar de decisão (o padrão do app é 0,3)
    NOME=CAMINHO     modelo ONNX extra, ex.: "Modelo antigo do app=C:/caminho/model_2026-09-09.onnx"
"""
import os
import platform
import sys
import time

import numpy as np
import onnxruntime as ort
import scipy.special
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from comum import ENTRADAS, FP32, INT8, MAX_LENGTH, MODELO_HF_DIR, VARIANTES, carregar_divisao

opcoes = [a for a in sys.argv[1:] if a.startswith("--limiar=")]
extras = [a for a in sys.argv[1:] if not a.startswith("--limiar=")]
LIMIAR = float(opcoes[-1].split("=", 1)[1]) if opcoes else 0.5
AQUECIMENTO = 10

_, _, df_test = carregar_divisao()
textos = df_test["text"].tolist()
y = df_test["label"].to_numpy()
n = len(textos)

tok = AutoTokenizer.from_pretrained(MODELO_HF_DIR)
enc = tok(textos, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="np")
enc = {k: enc[k].astype(np.int64) for k in ENTRADAS}


def medir(rodar):
    """Roda 1 mensagem por vez; retorna logits e tempo médio por mensagem (ms)."""
    for i in range(AQUECIMENTO):
        rodar(i)
    logits, tempos = [], []
    for i in range(n):
        t0 = time.perf_counter()
        logits.append(rodar(i))
        tempos.append(time.perf_counter() - t0)
    return np.concatenate(logits), 1000 * float(np.mean(tempos))


def pytorch():
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_HF_DIR).eval()

    def rodar(i):
        with torch.no_grad():
            return model(**{k: torch.from_numpy(enc[k][i:i + 1]) for k in ENTRADAS}).logits.numpy()
    return medir(rodar)


def onnx_modelo(caminho):
    sess = ort.InferenceSession(caminho, providers=["CPUExecutionProvider"])
    nomes = [x.name for x in sess.get_inputs()]
    return medir(lambda i: sess.run(None, {k: enc[k][i:i + 1] for k in nomes})[0])


modelos = [
    ("PyTorch FP32 (referência)", None),
    ("ONNX FP32 (sem quantizar)", FP32),
    ("INT8 dinâmico padrão", INT8),
    ("v1 per_channel (app)", os.path.join(VARIANTES, "v1_per_channel.onnx")),
    ("v2 per_channel+reduce_range", os.path.join(VARIANTES, "v2_per_channel_reduce_range.onnx")),
    ("v3 só MatMul (emb. FP32)", os.path.join(VARIANTES, "v3_so_matmul.onnx")),
    ("v4 INT8 excl. pooler+classif.", os.path.join(VARIANTES, "v4_excl_pooler_classifier.onnx")),
    ("v5 FP16", os.path.join(VARIANTES, "v5_fp16.onnx")),
]
modelos = [(nome, c) for nome, c in modelos if c is None or os.path.exists(c)]
modelos += [tuple(arg.split("=", 1)) for arg in extras]

print(f"teste: {n} mensagens (golpe={int(y.sum())}) | limiar {LIMIAR} | lote=1, max_length={MAX_LENGTH} | "
      f"CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} lógicos) | ort {ort.__version__}")
cab = (f"{'modelo':32} | {'MB':>6} | {'acur':>6} | {'prec':>6} | {'recall':>6} | {'F1':>6} | "
       f"{'FN':>3} | {'FP':>3} | {'ms/msg':>6} | {'≠PyTorch':>8}")
print(cab)
print("-" * len(cab))
p_ref = None
for nome, caminho in modelos:
    logits, ms = pytorch() if caminho is None else onnx_modelo(caminho)
    p = scipy.special.softmax(logits, axis=1)[:, 1]
    if p_ref is None:
        p_ref = p
    pred = (p >= LIMIAR).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred).ravel()
    arquivo = caminho or os.path.join(MODELO_HF_DIR, "model.safetensors")
    difs = int(((p_ref >= LIMIAR) != (p >= LIMIAR)).sum())
    print(f"{nome:32} | {os.path.getsize(arquivo) / 2**20:6.1f} | {accuracy_score(y, pred):.4f} | "
          f"{precision_score(y, pred, zero_division=0):.4f} | {recall_score(y, pred, zero_division=0):.4f} | "
          f"{f1_score(y, pred, zero_division=0):.4f} | {fn:>3} | {fp:>3} | {ms:6.1f} | {difs:>8}", flush=True)
