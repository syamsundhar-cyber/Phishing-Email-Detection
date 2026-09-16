\# 📧 Phishing Email Detection Model



A machine learning-based cybersecurity project that classifies emails as \*\*Safe\*\* or \*\*Phishing\*\* using textual content and security-related email features.



\## 🎯 Project Objective



The objective of this project is to develop a phishing email detection model capable of identifying potentially malicious or suspicious emails.



The model analyzes email text and additional security-related features such as:



\- URLs

\- Email addresses

\- Suspicious keywords

\- Special characters

\- Digits

\- Text length

\- Email formatting indicators



The system classifies each email into one of two categories:



\- \*\*0 → Safe / Legitimate\*\*

\- \*\*1 → Phishing\*\*



\---



\## 🛠️ Technologies Used



\- Python 3

\- Pandas

\- Scikit-learn

\- Matplotlib

\- Regular Expressions

\- TF-IDF

\- Logistic Regression

\- Sparse Feature Matrices



\---



\## 📊 Dataset



The project uses a public phishing email dataset containing multiple email sources.



The original dataset contained:



\- \*\*82,486 emails\*\*

\- \*\*39,595 Safe / Legitimate emails\*\*

\- \*\*42,891 Phishing emails\*\*



\### Dataset Preparation



The original dataset was cleaned before model training.



Cleaning steps included:



1\. Removing missing email text

2\. Removing empty emails

3\. Removing duplicate emails

4\. Validating classification labels

5\. Limiting extremely large email text

6\. Shuffling the dataset



After cleaning:



\- \*\*82,077 emails\*\* remained

\- \*\*39,233 Safe / Legitimate emails\*\*

\- \*\*42,844 Phishing emails\*\*



The original dataset is not included in this repository because of its large file size.



\---



\## 🔄 Project Workflow



```text

Email Dataset

&#x20;     ↓

Data Cleaning

&#x20;     ↓

Text Preprocessing

&#x20;     ↓

TF-IDF Feature Extraction

&#x20;     ↓

URL \& Security Feature Extraction

&#x20;     ↓

Feature Combination

&#x20;     ↓

Train/Test Split

&#x20;     ↓

Logistic Regression

&#x20;     ↓

Email Classification

&#x20;     ↓

Performance Evaluation

