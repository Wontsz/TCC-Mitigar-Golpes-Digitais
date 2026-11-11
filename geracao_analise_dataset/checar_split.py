import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/dataset_v7.csv')

_, X_test_b, _, _ = train_test_split(
    df['mensagem_chat_sintetica'], df['label_ia'],
    test_size=0.15, stratify=df['label_ia'], random_state=42)

tr, tmp = train_test_split(df, test_size=0.3, stratify=df['label_ia'], random_state=42)
val, test_ft = train_test_split(tmp, test_size=0.5, stratify=tmp['label_ia'], random_state=42)

print("baseline:", len(X_test_b), "| fine-tuning:", len(test_ft))
print("indices iguais:", set(X_test_b.index) == set(test_ft.index))