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
