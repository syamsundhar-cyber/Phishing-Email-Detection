# Phishing Email Detection Model

A machine learning-based cybersecurity project that analyzes email text and classifies emails as **Phishing** or **Safe** using text and email-related features.

## 🎯 Objective

The objective of this project is to develop a machine learning model that can identify potentially phishing emails by analyzing their content and detecting patterns commonly associated with suspicious messages.

The model uses:

- Email text
- URLs
- Email addresses
- Suspicious keywords
- Special characters
- Numbers and digits
- Uppercase characters
- Email length
- Punctuation patterns

## 🚀 Features

- Detects phishing and safe emails
- Uses TF-IDF for text feature extraction
- Extracts additional URL and keyword-based features
- Uses Logistic Regression for classification
- Displays accuracy, precision, recall, and F1-score
- Generates a confusion matrix
- Provides a classification report
- Supports testing a new email interactively
- Saves model evaluation results to a text file

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- TF-IDF
- Logistic Regression

## 📊 Dataset

The project uses the **Phishing Email Dataset** containing legitimate and phishing-related email messages.

Original dataset:

- Total emails: **82,486**
- Safe emails: **39,595**
- Phishing emails: **42,891**

### Data Preprocessing

The dataset was cleaned before training:

- Removed missing text and labels
- Removed empty email records
- Removed duplicate emails
- Validated class labels
- Limited extremely large email content
- Shuffled the dataset

After preprocessing:

- Final emails: **82,077**
- Safe emails: **39,233**
- Phishing emails: **42,844**

Class labels:

```text
0 = Safe / Legitimate
1 = Phishing
```
The original dataset files are not included in this repository because of their large file size.

## 🔍 Feature Extraction

### 1. TF-IDF Text Features

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert email text into numerical features.

The model uses:

- Unigrams
- Bigrams
- Up to 30,000 text features
- English stop-word removal

### 2. Additional Email Features

The model also extracts:

- Number of URLs
- Number of email addresses
- Suspicious keyword count
- Exclamation marks
- Question marks
- Dollar symbols
- Number of digits
- Email length
- Uppercase characters
- Special characters

These features are combined with the TF-IDF features before classification.

## 🤖 Machine Learning Model

The project uses **Logistic Regression** as the classification algorithm.

The dataset is divided using an **80/20 stratified train-test split**.

```text
80% → Training
20% → Testing
```

Random state:

```text
42
```

## 📈 Model Results

The trained model was evaluated on **16,416 testing emails**.

| Metric | Result |
|---|---:|
| Accuracy | **98.82%** |
| Precision | **98.70%** |
| Recall | **99.05%** |
| F1-Score | **98.88%** |

These results represent the performance on this particular dataset and train-test split.

## 📉 Confusion Matrix

The confusion matrix produced by the model is:

```text
[[7735  112]
 [  81 8488]]
```

Interpretation:

| Actual | Predicted | Count |
|---|---|---:|
| Safe | Safe | 7,735 |
| Safe | Phishing | 112 |
| Phishing | Safe | 81 |
| Phishing | Phishing | 8,488 |

The confusion matrix image is available in:

```text
confusion_matrix.png
```

## 🧪 Manual Email Test

The model was also tested with a new example email:

```text
Hi John, the meeting is scheduled for tomorrow at 10 AM.
Please review the attached agenda and let me know if you
have any questions.
```

Result:

```text
Prediction: SAFE
Confidence: 99.49%
```

The confidence value is a model probability estimate and should not be treated as a guarantee that an email is safe.

## 📁 Project Structure

```text
Phishing-Email-Detection/
│
├── phishing_detector.py
├── prepare_dataset.py
├── confusion_matrix.png
├── model_results.txt
├── README.md
└── .gitignore
```

The dataset files are kept locally and excluded from GitHub using `.gitignore`.

## ⚙️ Installation

Make sure Python is installed.

Install the required libraries:

```bash
python -m pip install pandas scikit-learn matplotlib
```

## ▶️ How to Run

### Step 1: Prepare the Dataset

Place the original dataset inside:

```text
archive/phishing_email.csv
```

Then run:

```bash
python prepare_dataset.py
```

This creates the cleaned:

```text
dataset.csv
```

### Step 2: Train and Test the Model

Run:

```bash
python phishing_detector.py
```

The program will:

1. Load the cleaned dataset
2. Clean the email text
3. Extract TF-IDF features
4. Extract additional email features
5. Train the Logistic Regression model
6. Evaluate the model
7. Display performance metrics
8. Generate the confusion matrix
9. Allow testing of a new email

## 📄 Output Files

After running the model, the following files are generated:

```text
confusion_matrix.png
model_results.txt
```

## 🔐 Cybersecurity Purpose

This project is designed for educational and defensive cybersecurity purposes.

It demonstrates how machine learning and email-content analysis can be used as one component of a phishing detection workflow.

Machine learning predictions should not be considered a replacement for security controls, email gateways, threat intelligence, or human verification.

## 🎓 Learning Outcomes

Through this project, I learned:

- Dataset cleaning and preprocessing
- Feature extraction using TF-IDF
- Combining text and numerical features
- Machine learning classification
- Logistic Regression
- Train-test splitting
- Model evaluation
- Confusion matrix analysis
- Precision, recall, and F1-score
- Basic phishing email detection techniques
- Python-based cybersecurity automation

## 👨‍💻 Author

**Syam Sundhar**

Cyber Security Intern

## 📌 Project

**Project Title:** Phishing Email Detection Model

**Category:** Cybersecurity / Machine Learning

**Purpose:** Educational and defensive security analysis
