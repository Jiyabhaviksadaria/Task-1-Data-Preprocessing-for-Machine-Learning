"""
==============================================================
  Task 1: Data Preprocessing for Machine Learning
==============================================================
Steps covered:
  1. Generate a raw dataset with intentional "messiness"
  2. Explore & understand the data
  3. Handle missing values
  4. Encode categorical variables
  5. Normalize / standardize numerical features
  6. Split into train / test sets
  7. Save processed artefacts
==============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# SECTION 1 – Generate a realistic raw dataset
# ──────────────────────────────────────────────────────────────
np.random.seed(42)
N = 300

raw_data = {
    # Numerical features (with missing values injected)
    "age":            np.where(np.random.rand(N) < 0.08,  np.nan, np.random.randint(18, 70, N).astype(float)),
    "salary":         np.where(np.random.rand(N) < 0.10,  np.nan, np.random.randint(25_000, 120_000, N).astype(float)),
    "years_exp":      np.where(np.random.rand(N) < 0.05,  np.nan, np.random.randint(0, 40, N).astype(float)),
    "hours_per_week": np.where(np.random.rand(N) < 0.07,  np.nan, np.random.randint(20, 60, N).astype(float)),
    "performance_score": np.where(np.random.rand(N) < 0.12, np.nan,
                                  np.round(np.random.uniform(1.0, 5.0, N), 1)),

    # Categorical features (with missing values)
    "gender":       np.where(np.random.rand(N) < 0.06,  None,
                             np.random.choice(["Male", "Female", "Other"], N, p=[0.48, 0.48, 0.04])),
    "department":   np.where(np.random.rand(N) < 0.04,  None,
                             np.random.choice(["Engineering", "Marketing", "HR", "Finance", "Sales"], N)),
    "education":    np.where(np.random.rand(N) < 0.05,  None,
                             np.random.choice(["High School", "Bachelor", "Master", "PhD"], N,
                                              p=[0.20, 0.45, 0.25, 0.10])),
    "employment_type": np.where(np.random.rand(N) < 0.03, None,
                                np.random.choice(["Full-time", "Part-time", "Contract"], N, p=[0.70, 0.20, 0.10])),

    # Target variable (binary classification: promoted or not)
    "promoted": np.random.choice([0, 1], N, p=[0.65, 0.35]),
}

df_raw = pd.DataFrame(raw_data)

print("=" * 60)
print("  SECTION 1 – Raw Dataset Overview")
print("=" * 60)
print(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns\n")
print(df_raw.head(8).to_string())
print()

# ──────────────────────────────────────────────────────────────
# SECTION 2 – Exploratory Data Analysis (EDA)
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 2 – Missing Value Analysis")
print("=" * 60)

missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]
print(missing_df.to_string())
print()

print("  Data Types & Unique Values")
print("-" * 40)
for col in df_raw.columns:
    dtype = df_raw[col].dtype
    n_unique = df_raw[col].nunique()
    print(f"  {col:<20} dtype={str(dtype):<10}  unique={n_unique}")
print()

print("  Numerical Summary")
print("-" * 40)
print(df_raw.describe().round(2).to_string())
print()

# ──────────────────────────────────────────────────────────────
# SECTION 3 – Handle Missing Values
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 3 – Handling Missing Values")
print("=" * 60)

df = df_raw.copy()

# --- 3a. Numerical columns: impute with median ---
num_cols = ["age", "salary", "years_exp", "hours_per_week", "performance_score"]
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])
print(f"  [Numerical] Imputed with MEDIAN: {num_cols}")

# --- 3b. Categorical columns: impute with most frequent ---
cat_cols = ["gender", "department", "education", "employment_type"]
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
print(f"  [Categorical] Imputed with MODE : {cat_cols}")

print(f"\n  Missing values remaining: {df.isnull().sum().sum()}")
print()

# ──────────────────────────────────────────────────────────────
# SECTION 4 – Encode Categorical Variables
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 4 – Encoding Categorical Variables")
print("=" * 60)

# --- 4a. Label Encoding for ordinal column (education) ---
education_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
df["education_encoded"] = df["education"].map(education_order)
print(f"  [Label Encoding – Ordinal] 'education'  →  education_encoded")
print(f"     Mapping: {education_order}")

# --- 4b. One-Hot Encoding for nominal columns ---
ohe_cols = ["gender", "department", "employment_type"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
print(f"\n  [One-Hot Encoding – Nominal] Columns encoded: {ohe_cols}")
new_ohe_cols = [c for c in df.columns if any(c.startswith(base + "_") for base in ohe_cols)]
print(f"     New columns created ({len(new_ohe_cols)}): {new_ohe_cols}")

# Drop original education column (now encoded)
df.drop(columns=["education"], inplace=True)

print(f"\n  Dataset shape after encoding: {df.shape}")
print()

# ──────────────────────────────────────────────────────────────
# SECTION 5 – Feature / Target Split
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 5 – Feature / Target Split")
print("=" * 60)

TARGET = "promoted"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"  Features (X): {X.shape[1]} columns")
print(f"  Target  (y): '{TARGET}'  |  Class distribution:")
print(f"     0 (Not promoted): {(y == 0).sum()}  ({(y==0).mean()*100:.1f}%)")
print(f"     1 (Promoted)    : {(y == 1).sum()}  ({(y==1).mean()*100:.1f}%)")
print()

# ──────────────────────────────────────────────────────────────
# SECTION 6 – Train / Test Split
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 6 – Train / Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 80 % train | 20 % test
    random_state=42,
    stratify=y             # preserve class ratio
)

print(f"  Train set : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test  set : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  Stratified split — class balance preserved ✓")
print()

# ──────────────────────────────────────────────────────────────
# SECTION 7 – Normalize / Standardize Numerical Features
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 7 – Feature Scaling")
print("=" * 60)

# Only scale the original numerical columns (not binary OHE columns)
scale_cols = num_cols + ["education_encoded"]

# --- 7a. StandardScaler (Z-score: mean=0, std=1) ---
std_scaler = StandardScaler()
X_train_std = X_train.copy()
X_test_std  = X_test.copy()
X_train_std[scale_cols] = std_scaler.fit_transform(X_train[scale_cols])   # fit on TRAIN only
X_test_std[scale_cols]  = std_scaler.transform(X_test[scale_cols])        # apply to TEST

# --- 7b. MinMaxScaler (0–1 range) – shown for comparison ---
mm_scaler = MinMaxScaler()
X_train_mm = X_train.copy()
X_test_mm  = X_test.copy()
X_train_mm[scale_cols] = mm_scaler.fit_transform(X_train[scale_cols])
X_test_mm[scale_cols]  = mm_scaler.transform(X_test[scale_cols])

print("  StandardScaler  →  mean=0, std=1  (best for algorithms like SVM, Logistic Regression, PCA)")
print("  MinMaxScaler    →  range [0, 1]   (best for Neural Networks, KNN, distance-based models)\n")

print("  Before scaling — Train set stats (numerical cols):")
print(X_train[scale_cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

print("\n  After StandardScaler — Train set stats:")
print(X_train_std[scale_cols].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

print("\n  After MinMaxScaler — Train set stats:")
print(X_train_mm[scale_cols].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
print()

# ──────────────────────────────────────────────────────────────
# SECTION 8 – Save processed datasets
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  SECTION 8 – Saving Processed Files")
print("=" * 60)

OUTPUT_DIR = "/mnt/user-data/outputs"

df_raw.to_csv(f"{OUTPUT_DIR}/raw_dataset.csv",              index=False)
X_train_std.to_csv(f"{OUTPUT_DIR}/X_train_scaled.csv",     index=False)
X_test_std.to_csv(f"{OUTPUT_DIR}/X_test_scaled.csv",       index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv",                 index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv",                   index=False)

print("  raw_dataset.csv        — original messy data")
print("  X_train_scaled.csv     — preprocessed train features (StandardScaler)")
print("  X_test_scaled.csv      — preprocessed test  features (StandardScaler)")
print("  y_train.csv            — train labels")
print("  y_test.csv             — test  labels")

print("\n" + "=" * 60)
print("  ✅  Preprocessing Pipeline Complete!")
print("=" * 60)
print(f"\n  Final feature matrix : {X_train_std.shape[1]} features")
print(f"  Train samples        : {X_train_std.shape[0]}")
print(f"  Test  samples        : {X_test_std.shape[0]}")
print("\n  Ready to feed into your ML model. 🚀")
