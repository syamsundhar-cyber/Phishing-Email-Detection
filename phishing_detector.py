import pandas as pd
import re
import os
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PHISHING EMAIL DETECTION MODEL
# ============================================================

DATASET_FILE = "dataset.csv"

print("=" * 70)
print("PHISHING EMAIL DETECTION MODEL")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/7] Loading dataset...")

df = pd.read_csv(
    DATASET_FILE,
    usecols=["text_combined", "label"]
)

df["text_combined"] = df["text_combined"].fillna("").astype(str)
df["label"] = df["label"].astype(int)

print(f"Total emails: {len(df):,}")


# ============================================================
# 2. CLEAN TEXT
# ============================================================

print("\n[2/7] Preparing email text...")

def clean_text(text):
    text = text.lower()

    # Replace URLs with a standard token.
    # The original dataset contains some normalized URLs,
    # so we also preserve URL-related words for feature extraction.
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["text_combined"].apply(clean_text)

# Remove any empty text remaining after cleaning
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

print(f"Emails after text cleaning: {len(df):,}")


# ============================================================
# 3. EXTRACT URL AND SUSPICIOUS KEYWORD FEATURES
# ============================================================

print("\n[3/7] Extracting URL and security-related features...")

suspicious_keywords = [
    "urgent",
    "verify",
    "verification",
    "account",
    "password",
    "login",
    "signin",
    "security",
    "confirm",
    "click",
    "update",
    "suspended",
    "suspension",
    "bank",
    "payment",
    "credit card",
    "debit card",
    "winner",
    "prize",
    "refund",
    "invoice",
    "limited time",
    "act now",
    "unsubscribe"
]


def extract_features(text):
    text_lower = text.lower()

    # URL indicators
    url_count = len(
        re.findall(
            r"https?://|www\.|http\s+www|https\s+www",
            text_lower
        )
    )

    # Email address count
    email_count = len(
        re.findall(
            r"\b[\w.+-]+@[\w.-]+\.\w+\b",
            text_lower
        )
    )

    # Number of suspicious keywords
    keyword_count = 0

    for keyword in suspicious_keywords:
        keyword_count += text_lower.count(keyword)

    # Character/format features
    exclamation_count = text.count("!")
    question_count = text.count("?")
    dollar_count = text.count("$")

    # Digit count
    digit_count = sum(char.isdigit() for char in text)

    # Text length
    text_length = len(text)

    # Number of uppercase characters in original text
    uppercase_count = sum(char.isupper() for char in text)

    # Number of special characters
    special_count = len(
        re.findall(r"[^a-zA-Z0-9\s]", text)
    )

    return [
        url_count,
        email_count,
        keyword_count,
        exclamation_count,
        question_count,
        dollar_count,
        digit_count,
        text_length,
        uppercase_count,
        special_count
    ]


feature_data = df["text_combined"].apply(extract_features)

numeric_features = pd.DataFrame(
    feature_data.tolist(),
    columns=[
        "url_count",
        "email_count",
        "suspicious_keyword_count",
        "exclamation_count",
        "question_count",
        "dollar_count",
        "digit_count",
        "text_length",
        "uppercase_count",
        "special_character_count"
    ]
)

print("Security-related features extracted:")
print(" - URL indicators")
print(" - Email address count")
print(" - Suspicious keyword count")
print(" - Special characters")
print(" - Digits")
print(" - Text length")
print(" - Other formatting indicators")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n[4/7] Splitting dataset into training and testing data...")

X_text = df["clean_text"]
X_numeric = numeric_features
y = df["label"]

X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_text,
    X_numeric,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training emails: {len(X_text_train):,}")
print(f"Testing emails : {len(X_text_test):,}")


# ============================================================
# 5. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n[5/7] Extracting TF-IDF text features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

print(f"TF-IDF training matrix: {X_train_tfidf.shape}")
print(f"TF-IDF testing matrix : {X_test_tfidf.shape}")


# ============================================================
# 6. COMBINE TEXT + NUMERIC FEATURES
# ============================================================

print("\n[6/7] Combining text and security features...")

scaler = MaxAbsScaler()

X_train_numeric = scaler.fit_transform(X_numeric_train)
X_test_numeric = scaler.transform(X_numeric_test)

X_train_final = hstack([
    X_train_tfidf,
    csr_matrix(X_train_numeric)
])

X_test_final = hstack([
    X_test_tfidf,
    csr_matrix(X_test_numeric)
])

print(f"Final training feature matrix: {X_train_final.shape}")
print(f"Final testing feature matrix : {X_test_final.shape}")


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print("\n[7/7] Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=500,
    solver="liblinear",
    random_state=42
)

model.fit(X_train_final, y_train)

print("Model training completed!")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test_final)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Safe", "Phishing"],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# Create confusion matrix image
fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(cm)

ax.set_title("Phishing Email Detection - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Safe", "Phishing"])
ax.set_yticklabels(["Safe", "Phishing"])

# Display values inside matrix
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=150
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

with open("model_results.txt", "w", encoding="utf-8") as file:

    file.write("PHISHING EMAIL DETECTION MODEL RESULTS\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Dataset size: {len(df):,}\n")
    file.write(f"Training emails: {len(X_text_train):,}\n")
    file.write(f"Testing emails: {len(X_text_test):,}\n\n")

    file.write("Classification:\n")
    file.write("0 = Safe / Legitimate\n")
    file.write("1 = Phishing\n\n")

    file.write(f"Accuracy : {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision * 100:.2f}%\n")
    file.write(f"Recall   : {recall * 100:.2f}%\n")
    file.write(f"F1-Score : {f1 * 100:.2f}%\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("Detailed Classification Report:\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["Safe", "Phishing"],
            zero_division=0
        )
    )

print("\nResults saved as:")
print("model_results.txt")


# ============================================================
# TEST A NEW EMAIL
# ============================================================

print("\n" + "=" * 70)
print("NEW EMAIL TEST")
print("=" * 70)

print("\nYou can test your own email text.")
print("Press ENTER without typing anything to skip this test.")

new_email = input("\nEnter email text: ").strip()

if new_email:

    cleaned_new_email = clean_text(new_email)

    new_text_tfidf = vectorizer.transform(
        [cleaned_new_email]
    )

    new_numeric = pd.DataFrame(
        [extract_features(new_email)],
        columns=numeric_features.columns
    )

    new_numeric_scaled = scaler.transform(new_numeric)

    new_final = hstack([
        new_text_tfidf,
        csr_matrix(new_numeric_scaled)
    ])

    prediction = model.predict(new_final)[0]

    probability = model.predict_proba(new_final)[0]

    if prediction == 1:
        result = "PHISHING"
        confidence = probability[1] * 100
    else:
        result = "SAFE"
        confidence = probability[0] * 100

    print("\nPrediction:", result)
    print(f"Model confidence: {confidence:.2f}%")

else:

    print("\nNew email test skipped.")


print("\n" + "=" * 70)
print("PROJECT EXECUTION COMPLETED")
print("=" * 70)