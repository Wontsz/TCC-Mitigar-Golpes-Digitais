"""Baseline TF-IDF + Regressão Logística (mesmos hiperparâmetros do baseline.py) em duas divisões:

(a) divisão do fine_tuning.py (70/15/15): treina só no treino, avalia no MESMO teste do BERTimbau;
(b) divisão antiga do baseline.py (85/15 única, antes do commit c3edfb8), para comparação.
Também verifica se os dois conjuntos de teste são iguais (como o checar_split.py).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from comum import CSV, carregar_divisao


def avaliar(nome, X_train, y_train, X_test, y_test):
    vec = TfidfVectorizer(max_features=500, lowercase=False)
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(X_train), y_train)
    pred = clf.predict(vec.transform(X_test))
    vn, fp, fn, vp = confusion_matrix(y_test, pred).ravel()
    print(f"{nome:44} | treino {len(X_train):4} | teste {len(X_test):3} | acur {accuracy_score(y_test, pred):.4f} | "
          f"prec {precision_score(y_test, pred):.4f} | recall {recall_score(y_test, pred):.4f} | "
          f"F1 {f1_score(y_test, pred):.4f} | FN {fn} | FP {fp}")


df_train, _, df_test = carregar_divisao()
avaliar("(a) divisão do fine_tuning (70/15/15)", df_train["text"], df_train["label"], df_test["text"], df_test["label"])

df = pd.read_csv(CSV)
X_tr, X_te, y_tr, y_te = train_test_split(
    df["mensagem_chat_sintetica"], df["label_ia"], test_size=0.15, stratify=df["label_ia"], random_state=42)
avaliar("(b) divisão antiga do baseline (85/15)", X_tr, y_tr, X_te, y_te)

comuns = len(set(X_te.index) & set(df_test.index))
print(f"\nconjuntos de teste iguais: {set(X_te.index) == set(df_test.index)} "
      f"({comuns} de {len(df_test)} mensagens em comum)")
