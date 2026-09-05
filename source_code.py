import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load Dataset
df = pd.read_csv("dataset/Womens Clothing E-Commerce Reviews.csv")

# 2. Create Sentiment
df = df.dropna(subset=["Review Text"])

df = df[df["Rating"] != 3]

df["Sentiment"] = df["Rating"].apply(
    lambda x: "Positive" if x >= 4 else "Negative"
)

# 3. Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Cleaned_Review"] = df["Review Text"].apply(preprocess_text)

# 4. Split Data
X = df["Cleaned_Review"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Evaluation
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="Positive"))
print("Recall:", recall_score(y_test, y_pred, pos_label="Positive"))
print("F1-Score:", f1_score(y_test, y_pred, pos_label="Positive"))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 8. Predict New Review
def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    review_tfidf = tfidf.transform([cleaned_review])

    prediction = model.predict(review_tfidf)[0]
    probability = model.predict_proba(review_tfidf).max()

    return prediction, probability


review = input("\nEnter a product review: ")

sentiment, confidence = predict_sentiment(review)

print("Predicted Sentiment:", sentiment)
print(f"Confidence: {confidence * 100:.2f}%")
