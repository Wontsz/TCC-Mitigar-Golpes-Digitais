from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

df = pd.read_csv('dataset_v2.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df['mensagem_chat_sintetica'], df['label_ia'], test_size=0.2,
    stratify=df['label_ia'], random_state=42
)


vec = TfidfVectorizer(max_features=500, lowercase=False)
Xt = vec.fit_transform(X_train)
Xte = vec.transform(X_test)


clf = LogisticRegression(max_iter=1000)
clf.fit(Xt, y_train)
preds = clf.predict(Xte)


print(f"Acurácia: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, target_names=['Legitima', 'Golpe']))