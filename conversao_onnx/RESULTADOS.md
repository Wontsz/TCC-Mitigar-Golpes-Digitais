# Resultados da conversão ONNX e do baseline

Medições de 13/10/2026, com as versões de `requirements.txt` (Python 3.13, Windows, CPU AMD Ryzen 5 7600, 12 threads lógicas).

## Conjunto de teste

- Dataset: `datasets/dataset_v7.csv`, com 3.396 mensagens.
- Divisão idêntica à do `fine_tuning.py`: 70/15/15, estratificada por `label_ia`, `random_state=42`.
  - Treino: 2.377 mensagens.
  - Validação: 509 mensagens.
  - **Teste: 510 mensagens**, sendo 232 golpes e 278 legítimas.
- Tokenização: tokenizer do próprio modelo, com `max_length=128` e padding até 128.
- Métricas de precisão, recall e F1 calculadas para a classe **golpe (1)**.

## Modelo fine-tunado × versões ONNX (limiar 0,5)

Cada modelo foi avaliado **uma mensagem por vez** (lote = 1), como o app faz. Script: `comparar_variantes.py`.

| Modelo | Tamanho | Acurácia | Precisão | Recall | F1 | Falsos negativos (golpes perdidos) | Falsos positivos | ms/mensagem |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 (referência) | 415,5 MB | 0,9569 | 0,9339 | 0,9741 | 0,9536 | 6 | 16 | 73,9 |
| ONNX INT8 dinâmico padrão | 104,6 MB | 0,9490 | 0,9558 | 0,9310 | 0,9432 | 16 | 10 | 23,4 |
| **ONNX INT8 v1 per-channel (no app)** | 105,0 MB | **0,9627** | 0,9419 | **0,9784** | **0,9598** | **5** | 14 | 23,9 |
| Modelo antigo do app (até 09/2026) | 104,6 MB | 0,9255 | 0,9076 | 0,9310 | 0,9191 | 16 | 22 | 20,0 |

- Tamanhos em MiB.
- Os tempos foram medidos no CPU de um desktop. No celular os valores absolutos serão maiores, mas a proporção entre os modelos serve de referência.
- O "modelo antigo do app" é o arquivo que estava em `app/src/main/assets/model.onnx` até a troca (SHA-256 `b20c317134d5e81710545c3a521291fd24332ec6beca6082efc0676d8849c0c8`). Ele não está versionado.

**Observação sobre o INT8.** A quantização dinâmica calcula a escala das ativações sobre o lote inteiro, então o resultado de um modelo INT8 muda um pouco conforme o tamanho do lote. Por isso a tabela acima usa lote = 1, igual ao app.

Exemplo: a v1, avaliada em lotes de 32 (`avaliar_modelos.py`), tem no limiar 0,5:
- recall de 0,9828 e 4 falsos negativos;
- contra 0,9784 e 5 falsos negativos com uma mensagem por vez.

Isso não afeta o PyTorch nem o ONNX FP32, que deram resultados idênticos aos do PyTorch (0 decisões diferentes).

A melhora da v1 sobre o PyTorch (5 contra 6 falsos negativos) está dentro do ruído da quantização: uma mensagem vale cerca de 0,4 ponto percentual de recall. Ela não deve ser lida como ganho real do modelo.

### No limiar padrão do app (0,3)

O app Guardião SMS usa **0,3 como limiar padrão de alerta**, ajustável nos Ajustes. Os mesmos modelos, uma mensagem por vez, no mesmo conjunto de teste:

| Modelo | Acurácia | Precisão | Recall | F1 | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|
| PyTorch FP32 (referência) | 0,9569 | 0,9303 | 0,9784 | 0,9538 | 5 | 17 |
| ONNX INT8 dinâmico padrão | 0,9451 | 0,9474 | 0,9310 | 0,9391 | 16 | 12 |
| **ONNX INT8 v1 per-channel (no app, limiar padrão)** | **0,9608** | 0,9380 | **0,9784** | **0,9578** | **5** | 15 |

Script: `python conversao_onnx/comparar_variantes.py --limiar=0.3`.

O `model.onnx` do app (SHA-256 abaixo), passado como modelo extra, deu exatamente o mesmo resultado da v1 gerada pelos scripts.

**Como interpretar:** na v1, o limiar 0,3 tem o **mesmo recall** do 0,5 (0,9784, 5 golpes perdidos) e **um falso positivo a mais** (15 contra 14). No PyTorch, o 0,3 detecta um golpe a mais que o 0,5. A diferença entre os limiares é de no máximo uma mensagem, porque o modelo dá probabilidades quase sempre próximas de 0 ou de 1.

### Outras variantes testadas (limiar 0,5, lote = 1)

| Variante | Tamanho | Acurácia | Recall | F1 | FN | FP | ms/mensagem |
|---|---|---|---|---|---|---|---|
| v2: per_channel + reduce_range | 105,0 MB | 0,9588 | 0,9784 | 0,9558 | 5 | 16 | 23,8 |
| v3: só MatMul (embeddings FP32) | 171,2 MB | 0,9490 | 0,9353 | 0,9435 | 15 | 11 | 23,5 |
| v4: INT8 sem pooler e classificador | 106,3 MB | 0,9490 | 0,9310 | 0,9432 | 16 | 10 | 23,6 |
| v5: FP16 | 208,0 MB | 0,9569 | 0,9741 | 0,9536 | 6 | 16 | 71,9 |

## Métricas por limiar (PyTorch FP32)

Também estão em `../tabela_thresholds.txt`. Script: `avaliar_modelos.py`.

| Limiar | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| 0,5 | 0,9569 | 0,9339 | 0,9741 | 0,9536 |
| 0,4 | 0,9569 | 0,9339 | 0,9741 | 0,9536 |
| 0,3 | 0,9569 | 0,9303 | 0,9784 | 0,9538 |

O limiar quase não muda o resultado, porque o modelo dá probabilidades muito próximas de 0 ou de 1.

## Modelo do app (v1) e reprodutibilidade

- Arquivo no app: `app/src/main/assets/model.onnx`, com 110.115.256 bytes.
- **SHA-256:** `d93dc8974fc2f45bf909e4b257dcf3e65062de040d28306c4f70af202f0a3463`
- Rodar `baixar_modelo.py`, `exportar_onnx.py` e `gerar_variantes.py` com as versões de `requirements.txt` gera `saida/variantes/v1_per_channel.onnx` com **exatamente esse hash**. Isso foi verificado em 13/10/2026.
- Exportação: exportador legado do PyTorch (`dynamo=False`), opset 17.
  - Entradas: `input_ids`, `attention_mask` e `token_type_ids` (int64).
  - Eixos dinâmicos: batch e sequência.
  - Saída: `logits`, com forma [batch, 2].
- Quantização: `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=True)`.

## Baseline (TF-IDF + Regressão Logística)

Os hiperparâmetros são os do `baseline.py`: `TfidfVectorizer(max_features=500, lowercase=False)` e `LogisticRegression(max_iter=1000)`. Script: `comparar_baseline.py`.

| Divisão | Treino | Teste | Acurácia | Precisão | Recall | F1 | FN | FP |
|---|---|---|---|---|---|---|---|---|
| (a) Divisão do `fine_tuning.py` (70/15/15), usada pelo `baseline.py` atual | 2.377 | 510 | **0,8843** | 0,8811 | 0,8621 | 0,8715 | 32 | 27 |
| (b) Divisão antiga do `baseline.py` (85/15 única) | 2.886 | 510 | 0,8784 | 0,8664 | 0,8664 | 0,8664 | 31 | 31 |

- Os dois conjuntos de teste **são diferentes**: só 269 das 510 mensagens são comuns. É o que o `checar_split.py` mostra ("indices iguais: False").
- Desde o commit `c3edfb8`, o `baseline.py` usa a divisão (a). Por isso o número dele, 0,8843, pode ser comparado diretamente com o do BERTimbau: 0,9569 em PyTorch e 0,9627 na v1 do app, no mesmo teste e no limiar 0,5.

## Como reproduzir

Na raiz do repositório:

```
pip install -r conversao_onnx/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python conversao_onnx/baixar_modelo.py
python conversao_onnx/exportar_onnx.py
python conversao_onnx/gerar_variantes.py
python conversao_onnx/comparar_variantes.py "Modelo antigo do app=<caminho do .onnx antigo>"   # o argumento é opcional
python conversao_onnx/comparar_variantes.py --limiar=0.3                                     # limiar padrão do app
python conversao_onnx/avaliar_modelos.py
python conversao_onnx/comparar_baseline.py
python conversao_onnx/gerar_referencia_tokenizer.py <saida.tsv>   # referência do teste do tokenizador do app
```
