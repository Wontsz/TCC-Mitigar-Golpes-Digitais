import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets import Dataset
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

# Divisão em Treino (70%) e Teste (30%)
df_train = df.sample(frac=0.7, random_state=42)
df_test = df.drop(df_train.index)

ds_train = Dataset.from_pandas(df_train)
ds_test = Dataset.from_pandas(df_test)

# Tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = ds_train.map(tokenize_function, batched=True)
test_dataset = ds_test.map(tokenize_function, batched=True)

# Weighting
# Calculamos os pesos inversamente proporcionais à frequência das classes
class_counts = df_train['label'].value_counts().to_dict()
# Atribuímos um peso maior à classe 1 (Golpe) para penalizar Falsos Negativos
weights = [1.0 / class_counts[0], 2.0 / class_counts[1]] 
class_weights = torch.tensor(weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

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
    num_train_epochs=10,             # ES vai cuidar
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_recall", # Não deixa passar golpes (foco)
    logging_dir='./logs',
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Interrompe se não melhorar em 2 épocas
)

print("Iniciando Fine-tuning com priorização de RECALL...")
trainer.train()

# rels
print("\n" + "="*30)
print("AVALIAÇÃO FINAL DO MODELO")
print("="*30)

preds_output = trainer.predict(test_dataset)
predictions = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids

print("\nMATRIZ DE CONFUSÃO:")
print(confusion_matrix(labels, predictions))

print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(labels, predictions, target_names=["Legitimo", "Golpe"]))

# Salva o modelo final
model.save_pretrained("./modelo_smishing_final")
tokenizer.save_pretrained("./modelo_smishing_final")
print("\n[SUCESSO] Modelo treinado e salvo!")