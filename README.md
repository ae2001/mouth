# mouth
Stop misinformation before it spreads from mouth to mouth.

# 🧠 VeriScope – Transformer-Based Fake News Detection System

> A production-ready AI system that detects and scores fake news content on social media using NLP, semantic claim verification, and propagation analysis.

---

## 📌 Overview

VeriScope is a full-stack system to detect **fake news in real-time**. It uses state-of-the-art **Transformer models**, **semantic similarity search**, and **graph analysis** to evaluate the **veracity of social media posts**.

The app flags posts as `True`, `False`, `Partly True`, or `Misleading`, and provides a confidence score, claim verification links, and bot-like activity signals.

---

## 🚀 Problem Statement

Misinformation spreads rapidly across social platforms, affecting elections, health, and society. Traditional moderation is not scalable. VeriScope solves this by:

- Automatically classifying text as **real or fake**
- Verifying claims against **trusted knowledge bases**
- Detecting **bot-like content propagation**
- Providing a **dashboard for high-risk content review**

---

## 🧠 Methodology

### 📘 Datasets Used

- **LIAR**: 12.8K labeled short political claims – [Link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- **FakeNewsNet**: Fake/real news with social context – [Link](https://github.com/KaiDMML/FakeNewsNet)
- **CREDBANK**: 60M tweets labeled by credibility
- **ClaimReview (Schema.org)**, Wikipedia, Snopes, PolitiFact for claim verification

---

### 🔍 Architecture Overview

```
+------------------+         +------------------------+  
|  Social Media    | ----->  |  Preprocessing Layer   |  
|  (Twitter/Reddit)|         +------------------------+  
        |                          ↓                  
        |               +-------------------------+  
        +-------------> | Transformer Classifier  | --> Label: True / False / Misleading  
                        +-------------------------+  
                                     ↓                    
                      +-------------------------------+  
                      | Semantic Claim Matcher (SBERT) |  
                      +-------------------------------+  
                                     ↓                    
                          +---------------------+  
                          | Bot Propagation Map |  
                          +---------------------+  
                                     ↓                    
                      +-------------------------------+  
                      | Risk Scoring & Explanation UI |  
                      +-------------------------------+  
```

---

## 🏗️ System Components

| Component              | Description                                                               |
|------------------------|---------------------------------------------------------------------------|
| `Transformer Model`    | Fine-tuned RoBERTa/BERT for fake news detection                          |
| `Claim Matcher`        | Sentence-BERT + FAISS to match claims with trusted sources               |
| `Bot Analyzer`         | Graph-based propagation & user behavior analysis                         |
| `Risk Scorer`          | Weighted fusion of model output, similarity, and propagation signal      |
| `Dashboard`            | Streamlit or Gradio app to review scores, matches, and flagged posts     |

---

## 🧪 Evaluation Metrics

| Module             | Metrics                     |
|--------------------|-----------------------------|
| Classification     | Precision, Recall, F1-Score |
| Claim Matching     | MRR, Cosine Similarity      |
| Bot Detection      | ROC-AUC, Accuracy           |
| End-to-End Score   | False Positive Rate, Coverage |

---

## 🛠️ Tech Stack

- **Language**: Python
- **NLP Models**: HuggingFace Transformers (BERT, RoBERTa), Sentence-BERT
- **Frameworks**: Streamlit / Gradio, FastAPI / Flask
- **Storage**: MongoDB, Elasticsearch, S3
- **Deployment**: Docker, AWS Lambda, Prometheus (Monitoring)
- **Optional**: Graph Neural Networks for user classification

---

## ⚙️ Folder Structure

```
veriscope-fake-news-detector/
├── app/                  # UI & API
├── data/                 # Datasets
├── models/               # Fine-tuned models
├── pipelines/            # Training, scoring, claim matching
├── utils/                # Preprocessing, API wrappers
├── configs/              # YAML configs
├── tests/                # Unit tests
├── monitoring/           # Logs, dashboards
├── notebooks/            # EDA, training experiments
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

## 💡 Future Work

- Multilingual support using `xlm-roberta`
- Image/meme fact-checking using multimodal models (CLIP, Flamingo)
- Continual learning from user feedback (active learning)
- Integration with social platform moderation systems

---

## 🧑‍💻 Maintainer

**Naveen Kumar Nallamothu**  
📍 Jersey City, NJ  
📧 Naveen.nallamothu3x@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
