# TCC Mitigação Golpes Digitais

Projeto acadêmico desenvolvido no Instituto Mauá de Tecnologia com foco na detecção de mensagens fraudulentas (smishing) utilizando Processamento de Linguagem Natural (PLN) e o modelo BERTimbau. O sistema busca identificar padrões de engenharia social em mensagens de texto em português brasileiro. Além da classificação automática de mensagens suspeitas, o projeto propõe a integração do modelo a um protótipo móvel capaz de fornecer alertas explicativos ao usuário em tempo real, promovendo conscientização digital e preservação da privacidade por meio do processamento local das mensagens.

Integrantes:
- João Pedro Souza Cruz
- Mateus Canotilho Wontroba
- Victor Checce Pregnolatto

Orientador: Prof. Dr. Evandro Catelani Ferraz

## Como reproduzir

- **Dataset:** o dataset final é `datasets/dataset_v7.csv` (colunas `mensagem_chat_sintetica` e `label_ia`, 1 = golpe). Os scripts (`fine_tuning.py`, `baseline.py`, `checar_split.py`, `analise_dataset.py`) devem ser rodados a partir da raiz do repositório.
- **Geração sintética com Gemini (`gerar_dataset_chat.py`):** a chave da API não fica no código. Defina a variável de ambiente `GEMINI_API_KEY` antes de rodar (PowerShell: `$env:GEMINI_API_KEY = "sua-chave"`). Se usar um arquivo `.env`, ele está no `.gitignore` e não deve ser versionado.
- **Conversão do modelo para ONNX (usado no app Android):** com Python 3.13, a partir da raiz:
  ```
  pip install -r conversao_onnx/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  python conversao_onnx/baixar_modelo.py        # baixa Wontsimt/detector-phishing-v3-final
  python conversao_onnx/exportar_onnx.py        # ONNX FP32 (opset 17) + INT8 dinâmico padrão
  python conversao_onnx/gerar_variantes.py      # variantes; a v1 (per_channel) é a do app
  python conversao_onnx/comparar_variantes.py   # métricas e tempo por mensagem (lote = 1)
  ```
  Os arquivos gerados ficam em `conversao_onnx/saida/` (fora do git). Resultados e SHA-256 do modelo do app em [`conversao_onnx/RESULTADOS.md`](conversao_onnx/RESULTADOS.md).

