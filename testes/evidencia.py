import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./modelo_smishing_final")
sess = ort.InferenceSession("modelo_onnx/model_int8.onnx")

mensagens = [
    "Detectamos um Pix nao autorizado. Confirme seus dados para cancelar: gg.gg/pix-cancelar",
    "Nubank: compra de R$ 54,90 aprovada no seu roxinho.",
    "Sua encomenda esta retida. Pague a taxa de liberacao: tiny.cc/taxa",
    "Oi mae, cheguei bem. Beijo!",
    "Clinica Odonto: lembrete da sua consulta amanha as 14:00.",
]

print(f"{'PROB':>9}  {'DECISAO':<9}  MENSAGEM")
for m in mensagens:
    enc = tok(m, padding="max_length", truncation=True, max_length=128, return_tensors="np")
    feeds = {i.name: enc[i.name].astype(np.int64) for i in sess.get_inputs()}
    logits = sess.run(None, feeds)[0][0]
    e = np.exp(logits - logits.max())
    p = float((e / e.sum())[1])
    print(f"{p*100:8.4f}%  {'GOLPE' if p >= 0.3 else 'LEGITIMO':<9}  {m}")