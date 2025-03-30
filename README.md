# 🛍️ Walmart Customer Review Sentiment Classifier (LLM Fine-Tuning)

This project fine-tunes a **BERT-based language model** on Walmart product reviews to classify customer sentiment as **Positive, Neutral, or Negative**.

We first trained the model on **all reviews**, and then fine-tuned it again using **only verified purchaser reviews** to assess the impact of data quality. The verified-only model produced more accurate and trustworthy predictions.



---

## 🚀 Project Goals

- 🔍 Extract **customer sentiment** from product reviews
- 🧹 Compare model performance on **all data vs. verified-only data**
- 🧠 Fine-tune a **Large Language Model (BERT)** for sentiment classification
- 📊 Visualize model confidence using softmax probabilities
- 🌐 (Next step) Deploy the model using **Streamlit** for interactive demo

---

## 📂 Dataset

Source: `WalmartProducts.csv`  
Features:
- `Review`: Text content of customer feedback
- `Rating`: 1–5 stars
- `Verified Purchaser`: Yes/No
- `Sentiment`: Derived from rating:
  - 1–2 → Negative
  - 3 → Neutral
  - 4–5 → Positive

---

## 🧠 Data Quality Matters: Full vs. Verified-Only Training

We conducted two rounds of fine-tuning:
1. **Full Dataset** – Model trained on all reviews  
2. **Verified-Only Dataset** – Model trained on reviews from verified purchasers only

> ✅ The verified-only model showed **higher accuracy and more consistent predictions**, indicating the importance of training on high-quality, trustworthy data.

---

## 🔧 Steps & Code Workflow

### 1. **Data Cleaning & Preprocessing**
- Removed reviews with missing text or titles
- Cleaned review text (lowercased, removed noise)
- Created a `Sentiment` column from ratings

### 2. **Encoding & Tokenization**
- Converted sentiment labels to numeric values
- Tokenized reviews using Hugging Face `bert-base-uncased`
- Applied padding/truncation to fixed length (128 tokens)

### 3. **Model Training**
- Fine-tuned `BertForSequenceClassification` for 3-class classification
- Used Hugging Face `Trainer` API
- Compared performance of full data model vs. verified-only model

### 4. **Evaluation & Confidence Visualization**
- Evaluated accuracy and model performance
- Implemented a prediction function with a **bar chart of model confidence** (softmax)

### 5. **(Upcoming) Streamlit Deployment**
We are building a **Streamlit app** that will allow users to:
- Paste a review
- See sentiment prediction
- Visualize model confidence in real time

---

## 📈 Results & Insights

| Model | Accuracy | Notes |
|-------|----------|-------|
| Full Dataset | ✅ Good | Some noise and misaligned text-labels observed |
| Verified Only | 🔥 Better | Cleaner training signal, improved precision on Negative sentiment |

> “Training on verified reviews led to better generalization, proving that **quality matters more than quantity** in real-world data.”

---

## 📌 Project Stack

| Tool | Use |
|------|-----|
| Hugging Face Transformers | LLM fine-tuning |
| PyTorch | Backend training |
| scikit-learn | Label encoding |
| Matplotlib | Confidence visualization |
| Streamlit (next step) | App deployment |
| Pandas | Data handling |

---



---
