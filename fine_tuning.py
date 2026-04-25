import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# 1. Configurações Iniciais
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
CSV_PATH = "dataset_mensagens_chat.csv"

# Carregamento e ajuste do Dataset
df = pd.read_csv(CSV_PATH)
# Mapeando text e label
df = df.rename(columns={
    'mensagem_chat_sintetica': 'text',
    'label_ia': 'label'
})

# Divisão simples em treino
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

ds_train = Dataset.from_pandas(df_train)
ds_test = Dataset.from_pandas(df_test)

# 2. Tokenização
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = ds_train.map(tokenize_function, batched=True)
tokenized_test = ds_test.map(tokenize_function, batched=True)

# 3. Carregamento do Modelo para Classificação Binária
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 4. Função de Métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# 5. Argumentos de Treino (Hiperparâmetros Justificados)
training_args = TrainingArguments(
    output_dir="./resultados_tcc",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,          # Valor padrão para não esquecer o treino (pré)
    per_device_train_batch_size=16,
    num_train_epochs=5,           # Conforme suegestão Evandro
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
)

# 6. Inicialização do Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("Iniciando Fine-tuning")
trainer.train()

model.save_pretrained("./modelo_smishing_final")
tokenizer.save_pretrained("./modelo_smishing_final")
print("\nSucesso")