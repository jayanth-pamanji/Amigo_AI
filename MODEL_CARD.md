# Part B: Sentiment Classifier

## Project Overview
This project implements a **sentiment classifier** for short text data using a simple baseline and an improved approach with a lightweight transformer. The goal is to predict sentiment labels (0 = negative, 1 = positive) for unseen test data while handling challenges like small dataset size, typos, and short text lengths.

---

## Input Data

- **Training set:** `data/sentiment/train.csv` (230 samples)  
- **Development set:** `data/sentiment/dev.csv` (50 samples)  
- **Test set:** `data/sentiment/test.csv` (50 samples; labels hidden)  

**Labels:**  
- 0 → Negative  
- 1 → Positive  

**Data handling notes:**  
- No modifications are made to the original `data/` directory.  
- For experimentation, copies can be created, but final scripts read from original location.  

---

## Baseline Approach

- **Model:** TF-IDF + Logistic Regression  
- **Text preprocessing:** Lowercasing, basic tokenization  
- **Performance on dev set:**

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.65  |
| Precision | 0.63  |
| Recall    | 0.64  |
| F1-score  | 0.63  |

- Observations: Baseline captures coarse sentiment patterns but struggles with typos and very short texts.

---

## Improved Approach

- **Model:** DistilBERT / MiniLM transformer (frozen layers + trainable classifier head)  
- **Text preprocessing:** Only basic tokenization and truncation/padding to 128 tokens; **emoji conversion using `emoji.demojize` was tried but showed no improvement**.  
- **Freezing strategy:** Most transformer layers frozen to reduce overfitting; only last layer + classifier head trainable  
- **Training:** Hugging Face `Trainer`, small batch size due to limited data, 10 epochs, learning rate 5e-5  

**Performance on dev set (without emoji improvement):**

| Metric          | Value |
|-----------------|-------|
| Accuracy        | 0.9167 |
| Precision (0)   | 0.9375 |
| Precision (1)   | 0.9062 |
| Recall (0)      | 0.8333 |
| Recall (1)      | 0.9667 |
| F1-score (0)    | 0.8824 |
| F1-score (1)    | 0.9355 |

**Confusion Matrix:**
```
[[15  3]
 [ 1 29]]
```

- Freezing most layers prevented overfitting on the small dataset.  
- Accuracy is slightly lower than the version where emojis were converted, indicating emoji conversion did not improve performance.

---

## Error Analysis

- **Example mistakes:**  
  1. Extremely short texts ("Ok") predicted incorrectly.  
  2. Texts with typos occasionally misclassified.  
  3. Code-switching or mixed-language texts misclassified occasionally.  

- **Hypotheses:**  
  - Model lacks enough examples for rare tokens or short ambiguous texts.  
  - Emoji conversion did not provide additional context for sentiment.

---

## Fairness & Robustness Observations

- Model performs well on standard text but may misclassify due to typos or unusual phrasing.  
- Code-switching partially reduces performance.  
- Small dataset increases sensitivity to class imbalance.

---

## Outputs

- **Predictions:** `submissions/sentiment_test_predictions.csv`  
  - Format: `[text,label]`  
  - Label mapping: 0 = Negative, 1 = Positive  

- **Scripts:**  
  - `src/train.py` or equivalent notebook produces predictions.  

---

## Reproducibility

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train.py
```

3. Test predictions are automatically saved in `submissions/sentiment_test_predictions.csv`.

---

## Assumptions & Notes

- Small dataset; trade-off between model size and overfitting.  
- Emoji conversion was tested but showed no performance improvement.  
- Baseline approach provided for comparison; transformer used for improved results.  
- Any unclear aspects were assumed and documented.