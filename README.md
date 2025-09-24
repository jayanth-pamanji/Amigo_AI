# Project README

## Overview
This repository contains two mini-projects implemented as part of the assignment:

1. **Retrieval-Augmented Generation (RAG) System**  
2. **Sentiment Classification**

Both projects are designed to run locally, use open-source models, and handle small datasets efficiently.

---

## Part A: Retrieval-Augmented Generation (RAG)

### Problem
The task is to answer questions based on a small custom document corpus using retrieval + generation.  
The dataset is **very small**, which makes retrieval and generation quality highly sensitive to design choices.

### Input
- `data/corpus/docs.jsonl`: Document corpus  
- `data/corpus/questions.json`: Questions file  
- `config.json`: Configuration file  

### Output
- `submissions/rag_answers.json`: Dictionary of answers in `{qid: answer}` format  
- `RAG_README.md`: Detailed documentation of the RAG system  

### Approaches Tested
1. **One document per chunk (baseline)** – Simple and efficient.  
2. **Combine by title** – Improved coherence for title-based queries.  
3. **Sliding window (200 tokens, overlap 50)** – Tried, but added complexity without clear gains.  
4. **Top-5 chunk retrieval** – Final choice, balancing context and conciseness.  

### Retrieval Choice
- Model: `all-MiniLM-L6-v2` (SentenceTransformers)  
- Similarity: Cosine similarity  
- Top 5 chunks retrieved as context  

### Local Model Choice
- Model: **TinyLlama-1.1B Chat**  
- Loaded with Hugging Face `transformers` pipeline  
- Optimized with `device_map="auto"` for GPU memory usage  

### Anti-Hallucination Measures
- Prompt restricted to **retrieved context only**  
- Short max tokens to avoid irrelevant details  
- Keyword inclusion in prompts  

### Failure Cases
- Missing keywords in generated answers  
- Verbose or truncated outputs  

---

## Part B: Sentiment Classifier

### Problem
Predict binary sentiment labels (0 = negative, 1 = positive) for short text.  
Dataset size is **very small**, increasing risk of overfitting.

### Input Data
- Train: 230 samples  
- Dev: 50 samples  
- Test: 50 samples (labels hidden)  

### Approaches
1. **Baseline: TF-IDF + Logistic Regression**  
   - Accuracy: ~0.65  
   - Weak on typos and short texts  

2. **Improved: DistilBERT/MiniLM Transformer**  
   - Frozen layers to prevent overfitting  
   - Classifier head trained on top  
   - Accuracy: ~0.92 on dev set  

### Error Analysis
- Struggles with:
  - Extremely short texts  
  - Typos and noisy text  
  - Code-switching  

### Fairness & Robustness
- Works well on clean English text  
- Sensitive to dataset imbalance and rare tokens  

### Outputs
- Predictions: `submissions/sentiment_test_predictions.csv`  

---

## Assumptions
- All tasks run **locally without API calls**  
- Small dataset size requires **lightweight models**  
- Final answers/scripts should **not modify original data**  

## Difficulties
- Very small dataset → harder to generalize  
- Missing keywords in retrieval/generation  
- Typos and code-switching in sentiment data  

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Reproducibility
1. Run RAG system:
   ```bash
   python src/run_rag.py
   ```
   → Outputs `submissions/rag_answers.json`

2. Run sentiment classifier:
   ```bash
   python src/train_sentiment.py
   ```
   → Outputs `submissions/sentiment_test_predictions.csv`
