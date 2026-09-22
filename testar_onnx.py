import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./modelo_smishing_final")
sess = ort.InferenceSession("modelo_onnx/model_int8.onnx")

print("Entradas do grafo:", [i.name for i in sess.get_inputs()])
print("Saidas do grafo:", [o.name for o in sess.get_outputs()])

mensagens = [
    "Seu CPF foi bloqueado. Regularize agora: bit.ly/cpf-reg",
    "Oi mae, cheguei bem. Beijo!",
    "URGENTE: sua conta sera encerrada. Acesse tiny.cc/banco",
    "Nubank: compra de R$ 54,90 aprovada no seu roxinho.",
]

for m in mensagens:
    enc = tok(m, padding="max_length", truncation=True, max_length=128, return_tensors="np")
    feeds = {i.name: enc[i.name].astype(np.int64) for i in sess.get_inputs()}
    logits = sess.run(None, feeds)[0][0]
    e = np.exp(logits - logits.max())
    p = (e / e.sum())[1]
    print(f"{p:.1%}  {'GOLPE' if p >= 0.3 else 'LEGITIMO'}  |  {m[:45]}")