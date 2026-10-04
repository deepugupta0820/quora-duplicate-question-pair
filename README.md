# Duplicate Question Detection Using NLP and ANN

## Overview

This project is an end-to-end **Duplicate Question Detection System** that determines whether two given questions are semantically similar.

The project uses the **Quora Question Pairs Dataset** and combines:

- Natural Language Processing (NLP)
- Text preprocessing
- Feature engineering
- Gensim Word2Vec embeddings
- Similarity features
- Artificial Neural Network (ANN)
- FastAPI for model serving
- Pydantic for input validation
- Streamlit for the user interface
- Docker for containerization

The system takes two questions as input and predicts whether they are **Duplicate** or **Not Duplicate**.

---

## Dataset

The project uses the **Quora Question Pairs Dataset**.

The dataset contains the following columns:

- `id` - Unique identifier for each question pair
- `qid1` - ID of the first question
- `qid2` - ID of the second question
- `question1` - First question
- `question2` - Second question
- `is_duplicate` - Target label
  - `1` → Duplicate
  - `0` → Not Duplicate

For model development, **100,000 question pairs** were used.

---

## Data Preprocessing

The following NLP preprocessing techniques were applied:

- Lowercasing
- Removing leading and trailing spaces
- HTML tag removal using BeautifulSoup
- URL removal
- Special character removal
- Contraction expansion
- Stopword removal
- Tokenization
- POS tagging
- WordNet Lemmatization

The same preprocessing pipeline is used during model inference.

---

## Feature Engineering

Multiple numerical features were extracted from each pair of questions.

### Text-Based Features

- Question length
- Word count
- Common words
- Total words
- Word share ratio
- First word match
- Last word match

### Similarity Features

- Jaccard similarity
- Fuzzy matching features
- Fuzz Ratio
- Partial Ratio

### Word2Vec Features

Gensim **Word2Vec with CBOW architecture** was used to generate semantic representations of questions.

Each question is converted into a **150-dimensional vector**.

Additional similarity features include:

- Cosine Similarity
- Euclidean Distance

---

## Model Architecture

The classification model was built using **Keras Sequential API**.

Architecture:

- Input Layer - 316 features
- Dense Layers
- ReLU Activation
- Dropout
- Batch Normalization
- Sigmoid Output Layer

The Sigmoid output represents the probability that two questions are duplicates.

---

## Model Training and Evaluation

The model was trained on **100,000 rows for 100 epochs**.

| Metric | Score |
|---|---|
| Training Accuracy | 0.87 |
| Validation Accuracy | 0.79 |
| ROC-AUC Score | 0.88 |
| F1 Score | 0.72 |

The model achieved a **ROC-AUC score of 0.88**.

---

## Model Artifacts

The trained components are stored inside the `model` directory.

```text
model/
├── model.pkl
├── scaler.pkl
└── word2vec.model
```
### `model.pkl`

Contains the trained ANN model used for prediction.

### `scaler.pkl`

Contains the fitted `StandardScaler` used to transform the 316 features before prediction.

### `word2vec.model`

Contains the trained Word2Vec model used to convert new questions into 150-dimensional vectors.

The same trained artifacts are reused during API inference to maintain consistency with the training pipeline.

---

# FastAPI Backend

FastAPI is used to expose the trained model as a REST API.

The API accepts two questions and returns the prediction.

### API Endpoint

```http
POST /predict
```

---

# Pydantic Validation

Pydantic is used to validate incoming API data before it reaches the ML pipeline.

This prevents invalid input from being passed to the prediction pipeline.

---

# FastAPI Prediction Pipeline

The complete inference pipeline is:

```text
User Input
    ↓
Pydantic Validation
    ↓
Text Preprocessing
    ↓
Word2Vec Embeddings
    ↓
Feature Engineering
    ↓
316 Features
    ↓
StandardScaler
    ↓
ANN Model
    ↓
Prediction Probability
    ↓
Duplicate / Not Duplicate
```

---

# Streamlit UI

A Streamlit frontend is provided for interacting with the model through a web interface.

The Streamlit application sends the questions to the FastAPI backend and displays: Duplicate / Not Duplicate prediction and Prediction probability

The architecture is:

```text
Streamlit UI
     ↓
FastAPI API
     ↓
Predictor
     ↓
Word2Vec + Scaler + ANN
     ↓
Prediction
```

---

# Project Structure

```text
quora-duplicate-question/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── predictor.py
│  
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── word2vec.model
│
├── streamlit_UI.py
├── requirements.txt
├── Dockerfile
├── questions.csv
├── Quora_duplicate_question.ipynb
└── README.md
```

---

# Running the Project Locally

## 1. Clone Repositry

```bash
git clone https://github.com/deepugupta0820/quora-duplicate-question-pair.git
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Start FastAPI

From the project root:
```bash
uvicorn app.main:app --reload
```
The API will run at:
```bash
http://127.0.0.1:8000
```
Swagger documentation:
```bash
http://127.0.0.1:8000/docs
```


## 4. Start Streamlit

Open another terminal:
```bash
streamlit run streamlit_app.py
```
The Streamlit UI will normally be available at:
```bash
http://localhost:8501
```

# Running the Project with Docker

## 1. Pull Docker Image

```bash
docker pull deepu0820/quora-duplicate-api
```

## 2. Run Docker Container

```bash
docker run -p 8000:8000 deepu0820/quora-duplicate-api
```
The API will run at:
```bash
http://localhost:8000
```
Swagger documentation:
```bash
http://localhost:8000/docs
```