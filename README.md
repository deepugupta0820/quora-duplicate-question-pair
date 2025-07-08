# Duplicate_Question_detection_using_NLP_and_ANN

## Overview
This project builds a deep learning model to detect whether two given questions are semantically similar (duplicates). For this we use Quora Question Pairs dataset. We apply extensive Natural Language Processing (NLP), feature engineering, word embeddings, and an Artificial Neural Network (ANN) for classification.

## Dataset Exploration
We use Quora Question Pairs dataset. It contains 6 columns:
- `id` for each question pairs and `qid1`, `qid2` for question1 and qestion2 
- `question1`, `question2`: two question texts, `is_duplicate`: label (1 if duplicate, 0 otherwise)
-  Examining the dataset for structure, missing values, and basic statistics.

## Data Preprocessing
- Lowercasing and trimming
- HTML tag removal (BeautifulSoup), Special character and URL removal
- Contractions expansion
- Stopword removal
- Lemmatization using WordNet and POS tagging

## Feature Engineering
- Text lengths, Word counts, Common and total words, Word share ratio
- Jaccard similarity
- Fuzzy matching features (fuzz_ratio, partial_ratio, etc.)
- First/last word match
- Used Gensim Word2Vec (CBOW architecture) to convert questions into vectors and calculated:
    - Cosine similarity
    - Euclidean distance

## Model Architecture
Built using Keras Sequential
   - Input layer (316 features)
   - Dense → ReLU → Dropout → BatchNorm layers
   - Output layer (Sigmoid for binary classification)

## Model Training and Evaluation
- The model was trained on 100,000 rows of data for 100 epochs.
- **Training Accuracy**: 0.87
- **Validation Accuracy**: 0.79
- **ROC AUC Score**: 0.88
- **F1 Score**: 0.72

## Conclusion
The ANN classifier showed good performance on the test dataset with balanced precision, recall, and F1 scores.So this project presents a robust model capable of detecting duplicate question pairs effectively.
