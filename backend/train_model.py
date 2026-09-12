from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sample_churn.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = ["gender","senior_citizen","partner","dependents","tenure","phone_service","internet_service","monthly_charges","total_charges","contract","payment_method"]
cat_cols = ["gender","partner","dependents","phone_service","internet_service","contract","payment_method"]
num_cols = ["senior_citizen","tenure","monthly_charges","total_charges"]
X, y = df[features], df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.22, random_state=42, stratify=y)
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),("num", StandardScaler(), num_cols)])
model = Pipeline([("preprocess", pre),("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42))])
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:,1]
pred = (proba >= .5).astype(int)
metrics = {"accuracy": round(float(accuracy_score(y_test,pred)),4),"roc_auc": round(float(roc_auc_score(y_test,proba)),4),"training_rows": len(X_train),"test_rows": len(X_test),"note":"Metrics are from the included synthetic demo dataset, not a production benchmark."}
joblib.dump(model, MODEL_DIR / "churn_model.pkl")
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Model saved:", MODEL_DIR / "churn_model.pkl")
print(metrics)
