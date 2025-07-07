import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

# === Load and Preprocess Data ===
df = pd.read_csv("data.csv")
df.drop(columns=["id", "Unnamed: 32"], errors='ignore', inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# === EDA ===
df.describe().to_csv("summary_statistics.csv")

sns.countplot(x='diagnosis', data=df)
plt.title("Class Distribution (0 = Benign, 1 = Malignant)")
plt.savefig("class_distribution.png")
plt.clf()

plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.clf()

# === Data Preparation ===
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X_scaled_df)
X_vt_df = pd.DataFrame(X_vt, columns=X.columns[vt.get_support()])

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_vt_df, y, test_size=0.3, stratify=y, random_state=0)

# === Models ===
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "SVM (Linear)": SVC(kernel='linear', probability=True),
    "Neural Net": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0)
}

results = []
feature_summary = {}

# === Model Loop ===
for name, model in models.items():
    model_results = {"Model": name}

    # --- Before Feature Selection ---
    start = time.time()
    model.fit(X_train_all, y_train)
    y_train_pred = model.predict(X_train_all)
    y_test_pred = model.predict(X_test_all)
    y_train_prob = model.predict_proba(X_train_all)[:, 1]
    y_test_prob = model.predict_proba(X_test_all)[:, 1]
    end = time.time()

    model_results.update({
        "Before FS Train Acc": accuracy_score(y_train, y_train_pred),
        "Before FS Test Acc": accuracy_score(y_test, y_test_pred),
        "Before FS Train AUC": roc_auc_score(y_train, y_train_prob),
        "Before FS Test AUC": roc_auc_score(y_test, y_test_prob),
        "Before FS Time": round(end - start, 4)
    })

    # --- ROC Curve Before FS ---
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_test_prob):.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name} (Before FS)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_curve_{name.replace(' ', '_').replace('(', '').replace(')', '')}_before_fs.png")
    plt.clf()

    # --- Skip FS if not supported ---
    if name in ["SVM (RBF)", "Neural Net"]:
        model_results.update({
            "After FS Train Acc": "N/A",
            "After FS Test Acc": "N/A",
            "After FS Train AUC": "N/A",
            "After FS Test AUC": "N/A",
            "After FS Time": "N/A",
            "Selected Features": "Not Applicable"
        })
        results.append(model_results)
        continue

    # --- Feature Selection ---
    selector = SelectFromModel(model)
    selector.fit(X_train_all, y_train)
    selected_cols = X_train_all.columns[selector.get_support()]
    feature_summary[name] = selected_cols.tolist()

    X_train_fs = selector.transform(X_train_all)
    X_test_fs = selector.transform(X_test_all)

    # --- After FS ---
    start = time.time()
    model.fit(X_train_fs, y_train)
    y_train_pred_fs = model.predict(X_train_fs)
    y_test_pred_fs = model.predict(X_test_fs)
    y_train_prob_fs = model.predict_proba(X_train_fs)[:, 1]
    y_test_prob_fs = model.predict_proba(X_test_fs)[:, 1]
    end = time.time()

    model_results.update({
        "After FS Train Acc": accuracy_score(y_train, y_train_pred_fs),
        "After FS Test Acc": accuracy_score(y_test, y_test_pred_fs),
        "After FS Train AUC": roc_auc_score(y_train, y_train_prob_fs),
        "After FS Test AUC": roc_auc_score(y_test, y_test_prob_fs),
        "After FS Time": round(end - start, 4),
        "Selected Features": selected_cols.tolist()
    })

    # --- ROC Curve After FS ---
    fpr, tpr, _ = roc_curve(y_test, y_test_prob_fs)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_test_prob_fs):.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name} (After FS)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"roc_curve_{name.replace(' ', '_').replace('(', '').replace(')', '')}_after_fs.png")
    plt.clf()

    results.append(model_results)

# === Save Outputs ===
results_df = pd.DataFrame(results)
results_df.to_csv("final_project_results_before_after_fs.csv", index=False)

feature_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in feature_summary.items()]))
feature_df.to_csv("selected_features_per_model.csv", index=False)

# Final Print
print("\n=== Final Model Comparison (Before vs After FS) ===")
print(results_df)
