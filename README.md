# 📧 Spam or Ham Detection

A machine learning project that classifies emails as **Spam** or **Ham (not spam)** using **TF-IDF vectorization** and an **XGBoost classifier**, with hyperparameter tuning via `GridSearchCV` and model interpretability powered by **SHAP**.

---

## ✨ Features

- 📊 **Exploratory data analysis** — inspects class distribution and confirms a real-world imbalance (~1:5 spam-to-ham ratio)
- 🔤 **TF-IDF vectorization** of raw email text (English stop words removed)
- 🌳 **XGBoost classifier** wrapped in a `scikit-learn` `Pipeline` alongside the vectorizer
- ⚖️ **Class imbalance handling** via XGBoost's `scale_pos_weight`
- 🔍 **Hyperparameter tuning** with `GridSearchCV` (5-fold CV, optimized for F1 score) across `learning_rate`, `max_depth`, and `subsample`
- 📈 **Performance evaluation** — F1 score, accuracy, and a confusion matrix with specificity/sensitivity breakdown
- 🧪 **Live predictions** on custom sample emails to sanity-check the model
- 🧠 **Model interpretability with SHAP**:
  - Global feature importance via summary plots
  - Per-email waterfall plots showing exactly which words pushed a prediction toward Spam or Ham

---

## 🛠️ Tech Stack

| Component            | Library                          |
|------------------------|------------------------------------|
| Data Handling           | `pandas` |
| Text Vectorization      | `scikit-learn` (`TfidfVectorizer`) |
| Model                   | `xgboost` (`XGBClassifier`) |
| Hyperparameter Tuning   | `scikit-learn` (`GridSearchCV`, `Pipeline`) |
| Evaluation              | `scikit-learn` (`f1_score`, `accuracy_score`, `confusion_matrix`) |
| Model Interpretability  | `shap` |
| Visualization           | `matplotlib` |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Anthronox-arch/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn xgboost shap matplotlib jupyter
```

### 3. Run the notebook
```bash
jupyter notebook Spam_or_Ham_Detection.ipynb
```

> 📌 Update the CSV path in the notebook to point to your local copy of `spam_or_not_spam.csv`.

---

## 📁 Dataset

- **Source file:** `spam_or_not_spam.csv`
- **Size:** ~3,000 emails
- **Columns:** `email` (raw text), `label` (0 = Ham, 1 = Spam)
- **Class balance:** roughly **1:5** — about 500 spam emails to 2,500 ham emails

---

## 📁 How It Works

1. **Load & clean** the dataset, dropping any rows with missing email text.
2. **Split** into train/test sets (80/20, stratified by label).
3. **Vectorize** email text with `TfidfVectorizer`.
4. **Handle class imbalance** by computing `scale_pos_weight` from the training set's spam-to-ham ratio and passing it to `XGBClassifier`.
5. **Tune hyperparameters** with `GridSearchCV` over `learning_rate`, `max_depth`, and `subsample`, scoring on F1 across 5 folds.
6. **Evaluate** the best model on the held-out test set — F1 score, accuracy, and a confusion matrix.
7. **Test on custom emails** to see real-time Spam/Ham predictions.
8. **Explain predictions** with SHAP — a summary plot for global word importance, and waterfall plots for individual emails.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Best CV F1 Score | **0.9567** |
| Test F1 Score | **0.9746** |
| Test Accuracy | **99.17%** |
| Specificity (True Negative Rate) | **99.80%** |
| Sensitivity (True Positive Rate) | **96.00%** |

**Best Parameters:** `learning_rate = 0.1`, `max_depth = 9`, `subsample = 0.7`

**Confusion Matrix** (600-email test set — 500 Ham / 100 Spam):

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 499 | 1 |
| **Actual Spam** | 4 | 96 |

---

## 🧩 Possible Improvements

- Deploy as an interactive **Streamlit** app for live email classification
- Experiment with word embeddings (e.g. Word2Vec, BERT) instead of TF-IDF
- Expand the dataset for more robust generalization
- Add threshold tuning to balance precision/recall for different use cases

---

## 👤 Author

**Muhammad Ahmad Awan**
BSCS Student, University of Management and Technology, Lahore
[GitHub: Anthronox-arch](https://github.com/Anthronox-arch)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
