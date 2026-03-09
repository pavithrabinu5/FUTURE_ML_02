"""
train_and_evaluate.py
----------------------
Full ML pipeline for Support Ticket Classification & Prioritization.

Steps:
  1. Load & explore data
  2. Text preprocessing
  3. Feature extraction (TF-IDF)
  4. Train category classifier
  5. Train priority classifier
  6. Evaluate both models
  7. Save models + generate report artifacts
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "support_tickets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from text_preprocessor import preprocess_series

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#4F46E5",
    "secondary": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
}
CAT_COLORS = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
PRI_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head(3).to_string())
print("\nCategory distribution:\n", df["category"].value_counts())
print("\nPriority distribution:\n", df["priority"].value_counts())


# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Support Ticket Dataset — Overview", fontsize=15, fontweight="bold", y=1.02)

# Category distribution
cat_counts = df["category"].value_counts()
axes[0].barh(cat_counts.index, cat_counts.values, color=CAT_COLORS[:len(cat_counts)])
axes[0].set_title("Tickets per Category")
axes[0].set_xlabel("Count")
for i, v in enumerate(cat_counts.values):
    axes[0].text(v + 5, i, str(v), va="center", fontsize=9)

# Priority distribution
pri_counts = df["priority"].value_counts().reindex(["High", "Medium", "Low"])
colors = [PRI_COLORS[p] for p in pri_counts.index]
axes[1].bar(pri_counts.index, pri_counts.values, color=colors, width=0.5)
axes[1].set_title("Tickets per Priority")
axes[1].set_xlabel("Priority Level")
axes[1].set_ylabel("Count")
for i, (label, v) in enumerate(zip(pri_counts.index, pri_counts.values)):
    axes[1].text(i, v + 5, str(v), ha="center", fontsize=10, fontweight="bold")

# Category × Priority heatmap
ct = pd.crosstab(df["category"], df["priority"])[["High", "Medium", "Low"]]
sns.heatmap(ct, ax=axes[2], annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white")
axes[2].set_title("Category × Priority Heatmap")
axes[2].set_xlabel("Priority")
axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ EDA chart saved.")

# Ticket length distribution
df["text_len"] = df["text"].str.len()
fig, ax = plt.subplots(figsize=(10, 4))
for cat, color in zip(df["category"].unique(), CAT_COLORS):
    subset = df[df["category"] == cat]["text_len"]
    ax.hist(subset, bins=30, alpha=0.6, label=cat, color=color)
ax.set_title("Ticket Text Length by Category")
ax.set_xlabel("Character Count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_text_length_dist.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ Length distribution chart saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Text Preprocessing")
print("=" * 60)

df["clean_text"] = preprocess_series(df["text"])
df["clean_len"] = df["clean_text"].str.len()

# Show before / after
for _, row in df.head(3).iterrows():
    print(f"\n[RAW]   {row['text'][:100]}...")
    print(f"[CLEAN] {row['clean_text']}")

# Remove empty rows post-cleaning
df = df[df["clean_text"].str.strip() != ""].copy()
print(f"\n✅ Preprocessing done. Clean corpus: {len(df)} tickets.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING (TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — TF-IDF Vectorization")
print("=" * 60)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=10000,
    sublinear_tf=True,    # apply log normalization
    min_df=2,
    max_df=0.95,
)

X = df["clean_text"]
y_cat = df["category"]
y_pri = df["priority"]

# Encode priority labels to ordinal (for analysis)
pri_order = {"High": 2, "Medium": 1, "Low": 0}
df["priority_num"] = df["priority"].map(pri_order)

# Train-test split (80/20, stratified)
X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test = train_test_split(
    X, y_cat, y_pri, test_size=0.2, random_state=42, stratify=y_cat
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Training set: {X_train_tfidf.shape}")
print(f"Test set:     {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")

# Show top TF-IDF terms
feature_names = np.array(tfidf.get_feature_names_out())
mean_tfidf = np.asarray(X_train_tfidf.mean(axis=0)).flatten()
top_idx = mean_tfidf.argsort()[-20:][::-1]
print("\nTop 20 TF-IDF terms:")
for term, score in zip(feature_names[top_idx], mean_tfidf[top_idx]):
    print(f"  {term:<30} {score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Model Selection (Category Classification)")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_tfidf, ycat_train,
                             cv=cv, scoring="f1_weighted", n_jobs=-1)
    comparison_results[name] = scores
    print(f"  {name:<25} F1={scores.mean():.4f} ± {scores.std():.4f}")

# Plot model comparison
fig, ax = plt.subplots(figsize=(10, 5))
names = list(comparison_results.keys())
means = [v.mean() for v in comparison_results.values()]
stds = [v.std() for v in comparison_results.values()]
colors_bar = [PALETTE["primary"] if m == max(means) else PALETTE["neutral"] for m in means]

bars = ax.barh(names, means, xerr=stds, color=colors_bar, capsize=5, height=0.5)
ax.set_xlim(0, 1.05)
ax.set_title("Model Comparison — 5-Fold CV F1-Weighted (Category)")
ax.set_xlabel("F1 Score (Weighted)")
for bar, m in zip(bars, means):
    ax.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{m:.4f}", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ Model comparison chart saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN FINAL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Training Final Models")
print("=" * 60)

# ── Category Classifier ──
cat_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
cat_model.fit(X_train_tfidf, ycat_train)
ycat_pred = cat_model.predict(X_test_tfidf)

cat_acc = accuracy_score(ycat_test, ycat_pred)
cat_f1 = f1_score(ycat_test, ycat_pred, average="weighted")
print(f"\n📊 Category Classifier (Logistic Regression)")
print(f"   Accuracy : {cat_acc:.4f}")
print(f"   F1-Score : {cat_f1:.4f}")
print(classification_report(ycat_test, ycat_pred))

# ── Priority Classifier ──
pri_model = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
pri_model.fit(X_train_tfidf, ypri_train)
ypri_pred = pri_model.predict(X_test_tfidf)

pri_acc = accuracy_score(ypri_test, ypri_pred)
pri_f1 = f1_score(ypri_test, ypri_pred, average="weighted")
print(f"\n📊 Priority Classifier (Logistic Regression — Balanced)")
print(f"   Accuracy : {pri_acc:.4f}")
print(f"   F1-Score : {pri_f1:.4f}")
print(classification_report(ypri_test, ypri_pred))


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — Confusion Matrices")
print("=" * 60)

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {title} saved.")

plot_confusion_matrix(
    ycat_test, ycat_pred,
    labels=sorted(df["category"].unique()),
    title="Confusion Matrix — Category Classifier",
    filename="04_confusion_category.png",
)

plot_confusion_matrix(
    ypri_test, ypri_pred,
    labels=["High", "Medium", "Low"],
    title="Confusion Matrix — Priority Classifier",
    filename="05_confusion_priority.png",
)


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLASS-WISE PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Class-Wise Performance Analysis")
print("=" * 60)

def plot_classwise(y_true, y_pred, title, filename, colors):
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    metric_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]]
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                      color=metric_colors[i], alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.legend()
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.4, label="0.8 threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {title} saved.")

plot_classwise(
    ycat_test, ycat_pred,
    title="Class-Wise Performance — Category Classifier",
    filename="06_classwise_category.png",
    colors=CAT_COLORS,
)

plot_classwise(
    ypri_test, ypri_pred,
    title="Class-Wise Performance — Priority Classifier",
    filename="07_classwise_priority.png",
    colors=[PRI_COLORS["High"], PRI_COLORS["Medium"], PRI_COLORS["Low"]],
)


# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE (Top TF-IDF terms per class)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9 — Top Predictive Terms per Category")
print("=" * 60)

feature_names = tfidf.get_feature_names_out()
classes = cat_model.classes_
coef = cat_model.coef_  # shape: (n_classes, n_features)

fig, axes = plt.subplots(1, len(classes), figsize=(20, 5))
fig.suptitle("Top 15 Predictive Terms per Category", fontsize=14, fontweight="bold")

for idx, (cls, ax) in enumerate(zip(classes, axes)):
    top_idx = np.argsort(coef[idx])[-15:]
    terms = feature_names[top_idx]
    scores = coef[idx][top_idx]
    color = CAT_COLORS[idx % len(CAT_COLORS)]

    ax.barh(terms, scores, color=color, alpha=0.85)
    ax.set_title(cls, color=color)
    ax.set_xlabel("Log-odds coefficient")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_top_terms_per_category.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✅ Feature importance chart saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10 — Saving Models")
print("=" * 60)

joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(cat_model, os.path.join(MODEL_DIR, "category_classifier.pkl"))
joblib.dump(pri_model, os.path.join(MODEL_DIR, "priority_classifier.pkl"))
print("✅ Models saved to /models/")


# ══════════════════════════════════════════════════════════════════════════════
# 11. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  FINAL RESULTS SUMMARY")
print("═" * 60)
print(f"  Dataset size          : {len(df):,} tickets")
print(f"  Training set          : {len(X_train):,} tickets")
print(f"  Test set              : {len(X_test):,} tickets")
print(f"  TF-IDF vocabulary     : {len(tfidf.vocabulary_):,} terms")
print(f"  Ngram range           : (1,2) — unigrams & bigrams")
print()
print(f"  ── Category Classifier ───────────────────────────────")
print(f"  Algorithm             : Logistic Regression")
print(f"  Accuracy              : {cat_acc:.2%}")
print(f"  Weighted F1           : {cat_f1:.2%}")
print()
print(f"  ── Priority Classifier ───────────────────────────────")
print(f"  Algorithm             : Logistic Regression (balanced)")
print(f"  Accuracy              : {pri_acc:.2%}")
print(f"  Weighted F1           : {pri_f1:.2%}")
print("═" * 60)
print("✅ All outputs saved to /outputs/")

# Save results dict for use by report generator
results = {
    "n_total": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "vocab_size": len(tfidf.vocabulary_),
    "cat_accuracy": round(cat_acc, 4),
    "cat_f1": round(cat_f1, 4),
    "pri_accuracy": round(pri_acc, 4),
    "pri_f1": round(pri_f1, 4),
    "cat_report": classification_report(ycat_test, ycat_pred),
    "pri_report": classification_report(ypri_test, ypri_pred),
}

import json
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("✅ Results saved to results.json")
