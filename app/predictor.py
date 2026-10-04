import os
import pickle
import re
import nltk
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# Same preprocessing order as the notebook.
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not",
    "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't": "need not",
    "oughtn't": "ought not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not",
    "that's": "that is", "there'd": "there would", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "won't": "will not",
    "would've": "would have", "wouldn't": "would not",
    "you're": "you are", "you've": "you have",
    "'ve": " have", "n't": " not", "'re": " are", "'ll": " will",
}


def replace_characters(q: str) -> str:
    q = str(q).lower().strip()
    q = q.replace("%", " percent").replace("$", " dollar ")
    q = q.replace("₹", " rupee ").replace("€", " euro ")
    q = q.replace("@", " at ").replace("[math]", "")
    q = q.replace(",000,000 ", "m ").replace(",000 ", "k ")
    q = re.sub(r"([0-9]+)000000000", r"\1b", q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)
    return q


def remove_contractions(q: str) -> str:
    return " ".join(CONTRACTIONS.get(word, word) for word in q.split())


def remove_html(q: str) -> str:
    return BeautifulSoup(q, "html.parser").get_text()


def remove_punc(q: str) -> str:
    return re.sub(re.compile(r"\W"), " ", q).strip()


def remove_url(q: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", q)


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith("J"):
        return wordnet.ADJ
    if pos_tag.startswith("V"):
        return wordnet.VERB
    if pos_tag.startswith("N"):
        return wordnet.NOUN
    if pos_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess(text: str) -> str:
    text = replace_characters(text)
    text = remove_contractions(text)
    text = remove_html(text)
    text = remove_punc(text)
    text = remove_url(text)

    # Same stopword logic as notebook.
    text = " ".join("" if word in STOP_WORDS else word for word in text.split())

    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    tokens = [LEMMATIZER.lemmatize(token, get_wordnet_pos(tag))
              for token, tag in tags]
    return " ".join(tokens)


def sentence_vector(sentence: str, w2v) -> np.ndarray:
    words = word_tokenize(sentence)
    # Equivalent intent to the notebook: keep vocabulary words and average them.
    words = [word for word in words if word in w2v.wv]
    if not words:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean([w2v.wv[word] for word in words], axis=0)


def cosine_similarity(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def build_features(q1: str, q2: str, w2v) -> np.ndarray:
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    q1_words = q1.split()
    q2_words = q2.split()

    q1_len = len(q1)
    q2_len = len(q2)
    q1_num_words = len(q1_words)
    q2_num_words = len(q2_words)

    common = len(set(q1_words) & set(q2_words))
    total = q1_num_words + q2_num_words
    word_share = round(common / total, 2) if total else 0.0

    s1, s2 = set(q1_words), set(q2_words)
    jaccard = len(s1 & s2) / max(len(s1 | s2), 1)

    fuzzy = [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]

    first_same = int(q1_words[0] == q2_words[0]) if q1_words and q2_words else 0
    last_same = int(q1_words[-1] == q2_words[-1]) if q1_words and q2_words else 0

    v1 = sentence_vector(q1, w2v)
    v2 = sentence_vector(q2, w2v)

    handcrafted = [
        q1_len, q2_len, q1_num_words, q2_num_words,
        common, total, word_share, jaccard,
        *fuzzy, first_same, last_same
    ]

    # Notebook feature order:
    # 16 engineered features + q1 150-vector + q2 150-vector
    features = np.array(
        handcrafted +
        v1.tolist() +
        v2.tolist() +
        [cosine_similarity(v1, v2), float(np.linalg.norm(v1 - v2))],
        dtype=np.float32
    )
    return features


class Predictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.w2v = None
        self.ready = False
        self.load()

    def load(self):
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        w2v_path = os.path.join(MODEL_DIR, "word2vec.model")

        missing = [p for p in [model_path, scaler_path, w2v_path] if not os.path.exists(p)]
        if missing:
            return

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.w2v = Word2Vec.load(w2v_path)
        self.ready = True

    def predict(self, question1: str, question2: str):
        if not self.ready:
            raise RuntimeError(
                "Model artifacts missing. Put model.pkl, scaler.pkl and "
                "word2vec.model inside the model/ directory."
            )

        X = build_features(question1, question2, self.w2v).reshape(1, -1)

        if X.shape[1] != 316:
            raise RuntimeError(f"Expected 316 features, got {X.shape[1]}")

        X_scaled = self.scaler.transform(X)
        probability = float(self.model.predict(X_scaled, verbose=0)[0][0])

        return {
            "is_duplicate": int(probability > 0.5),
            "probability": round(probability, 4),
            "prediction": "Duplicate" if probability > 0.5 else "Not Duplicate",
        }


predictor = Predictor()
