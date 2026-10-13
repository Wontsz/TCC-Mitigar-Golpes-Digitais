"""Passo 1: baixa o modelo fine-tunado (PyTorch/safetensors) do Hugging Face."""
from huggingface_hub import snapshot_download

from comum import MODELO_HF_DIR, MODELO_HF_ID

caminho = snapshot_download(MODELO_HF_ID, local_dir=MODELO_HF_DIR)
print("Modelo baixado em:", caminho)
