"""
Pre-compute SHAP values for XGBoost, LightGBM, Random Forest, Gradient Boosting
and save to reports/shap_cache.pkl so the app loads instantly.
"""
import shap, numpy as np, pandas as pd, joblib, pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))

BASE   = os.path.dirname(__file__)
MODELS = {
    "XGBoost":          "models/xgboost.pkl",
    "LightGBM":         "models/lightgbm.pkl",
    "Random Forest":    "models/random_forest.pkl",
    "Gradient Boosting":"models/gradient_boosting.pkl",
}
DATA   = os.path.join(BASE, "data", "raw", "students.csv")
CACHE  = os.path.join(BASE, "reports", "shap_cache.pkl")
N      = 300  # sample size

from src.preprocessor import _engineer_features
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")
df = pd.read_csv(DATA)
df_raw = df.drop(columns=["Target"], errors="ignore")
eng  = _engineer_features(df_raw.copy())
sc   = StandardScaler().fit(eng)
feat = list(eng.columns)

np.random.seed(42)
idx = np.random.choice(len(eng), min(N, len(eng)), replace=False)
X_sc  = sc.transform(eng)
X_df  = pd.DataFrame(X_sc, columns=feat)
X_s   = X_df.iloc[idx]

cache = {}
for name, path in MODELS.items():
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        print(f"  [skip] {name} — model fayli yo'q")
        continue
    print(f"Computing SHAP for {name}...", end=" ", flush=True)
    model = joblib.load(full)
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_s)
        cache[name] = {
            "shap_values":   sv,
            "feature_names": feat,
            "X_sample":      X_s.values,
            "sample_idx":    idx,
        }
        if isinstance(sv, list):
            print(f"OK (3-class, shape={np.array(sv).shape})")
        else:
            print(f"OK (shape={sv.shape})")
    except Exception as e:
        print(f"XATO: {e}")

with open(CACHE, "wb") as f:
    pickle.dump(cache, f)
print(f"\nSHAP cache saqlandi: {CACHE}  ({os.path.getsize(CACHE)//1024} KB)")
print("Endi app.py darrov SHAP natijalarini ko'rsatadi!")
