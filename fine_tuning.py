import pandas as pd
import numpy as np
import torch
from torch import nn
import scipy.special
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)

# Configs e Dados
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
CSV_PATH = "dataset_mensagens_chat.csv"

df = pd.read_csv(CSV_PATH).rename(columns={
    'mensagem_chat_sintetica': 'text', 
    'label_ia': 'label'
})

# Divisão em Treino (70%), Validação (15%) e Teste (15%) com estratificação
df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['label'], random_state=42)

ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)

# Tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = ds_train.map(tokenize_function, batched=True)
val_dataset = ds_val.map(tokenize_function, batched=True)
test_dataset = ds_test.map(tokenize_function, batched=True)

# Lógica de Pesos das Classes (Frequência base + Multiplicador para Recall)
classes = np.unique(df_train['label'])
pesos_base = compute_class_weight(class_weight='balanced', classes=classes, y=df_train['label'].values)

MULTIPLICADOR_RECALL = 2.0
weights = pesos_base.copy()
weights[1] = weights[1] * MULTIPLICADOR_RECALL
class_weights = torch.tensor(weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Métricas e Treinamento
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions)
    }

training_args = TrainingArguments(
    output_dir="./resultados_tcc",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_recall",
    logging_dir='./logs',
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Correção aqui
)

print("Iniciando Fine-tuning com priorização de RECALL...")
trainer.train()

# Avaliação final com múltiplos Thresholds
print("\n" + "="*30)
print("AVALIAÇÃO FINAL DO MODELO")
print("="*30)

preds_output = trainer.predict(test_dataset)
logits = preds_output.predictions
probabilidades = scipy.special.softmax(logits, axis=1)
prob_classe_golpe = probabilidades[:, 1]
labels = preds_output.label_ids

thresholds = [0.5, 0.4, 0.3]
linhas_tabela = []

for t in thresholds:
    print(f"\n--- THRESHOLD: {t} ---")
    preds_ajustadas = (prob_classe_golpe >= t).astype(int)
    
    print("MATRIZ DE CONFUSÃO:")
    print(confusion_matrix(labels, preds_ajustadas))
    
    print("RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(labels, preds_ajustadas, target_names=["Legitimo", "Golpe"]))
    
    # Coleta métricas da classe Golpe (1) para a tabela de resumo
    p = precision_score(labels, preds_ajustadas, zero_division=0)
    r = recall_score(labels, preds_ajustadas, zero_division=0)
    f1 = f1_score(labels, preds_ajustadas, zero_division=0)
    linhas_tabela.append(f"Threshold: {t} | Precisão: {p:.4f} | Recall: {r:.4f} | F1-Score: {f1:.4f}")

# Salva arquivo de texto com o resumo para o TCC
with open("tabela_thresholds.txt", "w") as f:
    f.write("\n".join(linhas_tabela))

# Salva o modelo final
model.save_pretrained("./modelo_smishing_final")
tokenizer.save_pretrained("./modelo_smishing_final")
print("\n[SUCESSO] Modelo treinado, resultados salvos em 'tabela_thresholds.txt' e modelo exportado!")