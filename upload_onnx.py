import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "victorcpregno/detector-smishing-v7"

for arquivo in ["model.onnx", "model_int8.onnx"]:
    print(f"Enviando {arquivo}...")
    api.upload_file(
        path_or_fileobj=f"modelo_onnx/{arquivo}",
        path_in_repo=f"onnx/{arquivo}",
        repo_id=repo,
    )
print("Concluido")