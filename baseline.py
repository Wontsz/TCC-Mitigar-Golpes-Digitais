from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

df = pd.read_csv('datasets/dataset_v7.csv')

# Mesma divisão do fine_tuning.py (70/15/15), para que o baseline seja
# avaliado exatamente no mesmo conjunto de teste do BERTimbau.
df_train, df_temp = train_test_split(
    df, test_size=0.3, stratify=df['label_ia'], random_state=42
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, stratify=df_temp['label_ia'], random_state=42
)

X_train, y_train = df_train['mensagem_chat_sintetica'], df_train['label_ia']
X_test, y_test = df_test['mensagem_chat_sintetica'], df_test['label_ia']

vec = TfidfVectorizer(max_features=500, lowercase=False)
Xt = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xt, y_train)
preds = clf.predict(Xte)

print(f"Acurácia: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, target_names=['Legitima', 'Golpe']))