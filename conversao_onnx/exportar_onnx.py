"""Passo 2: exporta o modelo para ONNX (exportador legado, opset 17) e quantiza em INT8 dinâmico padrão."""
import os

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from comum import ENTRADAS, FP32, INT8, MAX_LENGTH, MODELO_HF_DIR, SAIDA

os.makedirs(SAIDA, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODELO_HF_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODELO_HF_DIR)
model.eval()

enc = tok(["Seu CPF foi bloqueado. Regularize agora: bit.ly/cpf-reg", "Oi mae, cheguei bem."],
          padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
args = tuple(enc[k] for k in ENTRADAS)
eixos = {n: {0: "batch", 1: "sequence"} for n in ENTRADAS}
eixos["logits"] = {0: "batch"}


class Wrapper(torch.nn.Module):
    """Expõe só os logits, com as 3 entradas nomeadas que o app envia."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits


with torch.no_grad():
    torch.onnx.export(
        Wrapper(model), args, FP32,
        input_names=list(ENTRADAS), output_names=["logits"],
        dynamic_axes=eixos, opset_version=17, do_constant_folding=True,
        dynamo=False,
    )
onnx.checker.check_model(FP32)
print("FP32 exportado:", FP32, os.path.getsize(FP32), "bytes")

quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
onnx.checker.check_model(INT8)
print("INT8 quantizado:", INT8, os.path.getsize(INT8), "bytes")

m = onnx.load(INT8)
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
for i in m.graph.input:
    print("entrada:", i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("saida:", o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
