import pandas as pd
import os

# ============================================================
# PHISHING EMAIL DETECTION - DATA PREPARATION
# ============================================================

INPUT_FILE = os.path.join("archive", "phishing_email.csv")
OUTPUT_FILE = "dataset.csv"

print("=" * 60)
print("PHISHING EMAIL DETECTION - DATA PREPARATION")
print("=" * 60)

print("\nLoading dataset...")

# Load only the columns required for our project
df = pd.read_csv(
    INPUT_FILE,
    usecols=["text_combined", "label"]
)

print(f"Original rows: {len(df):,}")

# ------------------------------------------------------------
# 1. Remove missing values
# ------------------------------------------------------------

before = len(df)

df = df.dropna(subset=["text_combined", "label"])

print(f"Rows after removing missing values: {len(df):,}")
print(f"Removed: {before - len(df):,}")

# ------------------------------------------------------------
# 2. Remove empty email text
# ------------------------------------------------------------

before = len(df)

df["text_combined"] = df["text_combined"].astype(str).str.strip()

df = df[df["text_combined"] != ""]

print(f"Rows after removing empty emails: {len(df):,}")
print(f"Removed: {before - len(df):,}")

# ------------------------------------------------------------
# 3. Remove duplicate emails
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset=["text_combined"])

print(f"Rows after removing duplicate emails: {len(df):,}")
print(f"Duplicate emails removed: {before - len(df):,}")

# ------------------------------------------------------------
# 4. Keep only valid labels
# ------------------------------------------------------------

df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df[df["label"].isin([0, 1])]

df["label"] = df["label"].astype(int)

# ------------------------------------------------------------
# 5. Limit extremely large emails
# ------------------------------------------------------------

# Very large emails can consume unnecessary memory during
# TF-IDF processing. We keep the first 100,000 characters.

MAX_TEXT_LENGTH = 100000

df["text_combined"] = df["text_combined"].str.slice(0, MAX_TEXT_LENGTH)

# ------------------------------------------------------------
# 6. Shuffle the dataset
# ------------------------------------------------------------

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------------------------
# 7. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------------------
# 8. Display final statistics
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"\nFinal rows: {len(df):,}")

print("\nLabel distribution:")

label_counts = df["label"].value_counts().sort_index()

for label, count in label_counts.items():
    if label == 0:
        name = "Safe / Legitimate"
    else:
        name = "Phishing"

    print(f"  {label} ({name}): {count:,}")

print(f"\nOutput file: {OUTPUT_FILE}")

print("\nDataset is ready for machine learning.")

print("=" * 60)