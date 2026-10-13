"""Passo 3: gera variantes de quantização/conversão a partir do model_fp32.onnx.

A v1 (per_channel) é a que está no app Guardião SMS.
"""
import collections
import hashlib
import os

import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic

from comum import FP32, INT8, REPO, VARIANTES

os.makedirs(VARIANTES, exist_ok=True)

# O quantizador converte cada Gemm em MatMul+Add antes de aplicar a exclusão,
# então os nomes "..._Gemm_MatMul" também precisam estar na lista.
EXCLUIR_POOLER_CLASSIFICADOR = [
    "/m/bert/pooler/Gather",
    "/m/bert/pooler/dense/Gemm", "/m/bert/pooler/dense/Gemm_MatMul",
    "/m/classifier/Gemm", "/m/classifier/Gemm_MatMul",
]

variantes = {
    "v1_per_channel.onnx": dict(per_channel=True),
    "v2_per_channel_reduce_range.onnx": dict(per_channel=True, reduce_range=True),
    "v3_so_matmul.onnx": dict(op_types_to_quantize=["MatMul"]),
    "v4_excl_pooler_classifier.onnx": dict(nodes_to_exclude=EXCLUIR_POOLER_CLASSIFICADOR),
}
for nome, kw in variantes.items():
    quantize_dynamic(FP32, os.path.join(VARIANTES, nome), weight_type=QuantType.QInt8, **kw)
    print("ok", nome, kw)

m16 = float16.convert_float_to_float16(onnx.load(FP32), keep_io_types=True)
onnx.save(m16, os.path.join(VARIANTES, "v5_fp16.onnx"))
print("ok v5_fp16.onnx")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


print("\n--- arquivos gerados ---")
for p in [INT8] + [os.path.join(VARIANTES, f) for f in sorted(os.listdir(VARIANTES))]:
    onnx.checker.check_model(p)
    ops = collections.Counter(n.op_type for n in onnx.load(p).graph.node)
    print(f"{os.path.relpath(p, REPO)} | {os.path.getsize(p)} bytes | sha256 {sha256(p)} | "
          f"MatMulInteger={ops.get('MatMulInteger', 0)} MatMul={ops.get('MatMul', 0)}")
