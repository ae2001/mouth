# Veriscope: Probabilistic Fake News Detection System

## Overview

Veriscope is a real-time, probabilistic fake news detection system. It predicts the veracity of social media claims on a continuous scale from 1.0 (true) to 0.0 (false), combining fine-tuned transformer models and a meta learning layer enhanced with mathematically derived credibility features. This system is designed for high-stakes applications where binary classification is insufficient, such as political discourse, public health, and policy communication.

---

## Problem Statement

Traditional fake news classifiers rely on binary or multi-class labels, which fail to capture the subtle spectrum of misinformation. Furthermore, they ignore speaker credibility history and contextual factors. Veriscope addresses these limitations by predicting a probabilistic veracity score using ensemble learning, thereby offering a more accurate and flexible solution for misinformation detection.

---

## Dataset: LIAR

**Source**: LIAR dataset with 12,822 labeled claims

### Key Fields
- `claim`: Textual statement made by a public figure or entity
- `label`: Truthfulness rating (true, mostly-true, etc.)
- `speaker`, `party`, `state`: Metadata about the person/entity making the claim
- `true_count`, `false_count`, etc.: Speaker’s historical credibility data
- `context`: Medium of the statement (e.g., news release, tweet)

### Label Conversion

Labels are mapped to a continuous scale:

- `true` → 1.0  
- `mostly-true` → 0.8  
- `half-true` → 0.6  
- `barely-true` → 0.4  
- `pants-fire` → 0.2  
- `false` → 0.0  

These scores serve as regression targets for training models.

---

## Methodology

### Step 1: Data Preprocessing

- Clean text: Lowercasing, punctuation removal, whitespace normalization
- Normalize categorical fields (e.g., unknown for missing job titles)
- Map `label` column to probabilistic `veracity_score`

### Step 2: Mathematically Derived Features

Objective, numeric features are engineered from speaker credibility history:

**Credibility Score:**
\[
\text{credibility\_score} = \frac{1 \cdot \text{true} + 0.8 \cdot \text{mostly\_true} + 0.6 \cdot \text{half\_true} + 0.4 \cdot \text{barely\_true} + 0.2 \cdot \text{pants\_fire} + 0 \cdot \text{false}}{\text{total\_claims}}
\]

**Liar Index:**
\[
\text{liar\_index} = \frac{\text{false} + \text{pants\_fire}}{\text{total\_claims}}
\]

**False-to-True Ratio:**
\[
\text{false\_true\_ratio} = \frac{\text{false} + \text{pants\_fire}}{\text{true} + \varepsilon}, \quad \varepsilon = 10^{-5}
\]

**Entropy of Truthfulness:**
\[
H(p) = - \sum_{i} p_i \log(p_i), \quad p_i = \frac{\text{label\_count}_i}{\text{total\_claims}}
\]

These are later fed into the meta learner along with transformer predictions.

---

### Step 3: Base Models (Transformers as Regressors)

Five transformer models are fine-tuned as regressors:

- BERT-base
- RoBERTa-base
- DistilBERT
- ALBERT
- Longformer

Each model predicts a continuous veracity score using Mean Squared Error (MSE) as the loss function. Predictions on the validation set are stored for meta learning.

---

### Step 4: Meta Learner (Stacked Ensemble)

The meta learner takes five model predictions and engineered features as input:

**Input to Meta Model:**
\[
\textbf{x} = [s_{\text{BERT}}, s_{\text{RoBERTa}}, s_{\text{DistilBERT}}, s_{\text{ALBERT}}, s_{\text{Longformer}}, \text{credibility\_score}, \text{liar\_index}, \text{false\_true\_ratio}, \text{entropy}]
\]

**Meta Learner Models:**
- XGBoost Regressor (recommended)
- LightGBM Regressor
- Linear Regression (baseline)

The output is the final predicted veracity score for the claim.

---

## Inference Pipeline

1. Input: A new claim (string)
2. Preprocess the text
3. Generate predictions from each base model
4. Compute derived features from metadata (if available)
5. Feed predictions and features into the meta model
6. Output: A score in [0.0, 1.0] representing veracity

---

## Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Pearson and Spearman Correlation
- AUROC (for threshold-based classification)
- Coverage of flagged content below a risk threshold

---

## Folder Structure

veriscope/
├── data/
│ ├── raw/ # Original LIAR dataset
│ ├── processed/ # Cleaned and labeled dataset
│ └── meta_input.csv # Meta model training features
├── models/
│ ├── bert_model.pt
│ ├── roberta_model.pt
│ ├── distilbert_model.pt
│ ├── albert_model.pt
│ ├── longformer_model.pt
│ └── meta_model.pkl
├── train/
│ ├── train_bert.py
│ ├── train_roberta.py
│ ├── train_distilbert.py
│ ├── train_albert.py
│ ├── train_longformer.py
│ └── train_meta.py
├── inference/
│ └── ensemble_predict.py # Runs base models + meta model
├── utils/
│ ├── preprocess.py # Text and label processing
│ ├── feature_engineering.py # Derived features computation
│ └── evaluation.py # Metrics and plots
├── app/
│ └── main.py # Optional: Streamlit or Flask UI
├── README.md
├── requirements.txt
└── config.yaml

---

## Expected Output

Given a textual claim (e.g., from social media), the system will output:
- A veracity probability score between 0.0 and 1.0
- Optional: Individual model predictions
- Optional: Risk flags or alerts based on configured thresholds

---

## Deployment and Extensions

The system can be extended with:

- Claim matching: Embedding-based similarity search against Wikipedia, PolitiFact
- Network analysis: Identify bot-like propagation via graph modeling
- Feedback loop: Incorporate user or moderator feedback for retraining
- Frontend: Integrate with Streamlit/Gradio for live testing

---

## License

This project is open-source and available for research and educational use.
