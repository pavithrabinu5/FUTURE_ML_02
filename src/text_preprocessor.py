"""
text_preprocessor.py
---------------------
Handles all text cleaning and normalization for support tickets.
Uses only Python stdlib + scikit-learn (no NLTK/spaCy required).
"""

import re
import string

# Common English stopwords (stdlib subset — no NLTK needed)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "hi", "hello", "dear", "regards", "sincerely",
    "thanks", "thank", "please", "would", "could", "like", "get", "also",
    "us", "also", "may", "one", "also",
}


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove invoice/ticket numbers (noise)
    5. Remove punctuation & special characters
    6. Remove digits
    7. Remove extra whitespace
    8. Remove stopwords
    9. Remove very short tokens (< 2 chars)
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # 4. Remove invoice/ticket numbers (INV-xxx, TKT-xxx, #123)
    text = re.sub(r"(inv|tkt|ticket|invoice|#)[\s\-]?\w+", "", text)

    # 5. Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # 6. Remove digits
    text = re.sub(r"\d+", "", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Tokenize and remove stopwords + short tokens
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

    return " ".join(tokens)


def preprocess_series(series):
    """Apply clean_text to a pandas Series."""
    return series.apply(clean_text)


# ── Quick sanity test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Hi support team, I was charged twice for my subscription this month. Please refund immediately.",
        "Critical bug: all our user data appears corrupted after the last update. URGENT.",
        "Hello, Can you explain the $99.00 charge on Invoice INV-2024-0456?",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", clean_text(s))
        print()
