import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Keep important markers for sarcasm/negation
    tokens = [w for w in tokens if w not in stop_words or w in ['not', 'no', 'never', 'but', 'however', 'sure', 'right', 'wow', 'great', 'love']]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# 1. Load the Master Dataset
DATA_PATH = "master_sentiment_dataset.csv"
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found. Run generate_dataset.py first.")
    exit()

df_final = pd.read_csv(DATA_PATH)

print(f"Dataset Balance:")
print(df_final['label'].value_counts())
print(f"Total samples: {len(df_final)}")

# 2. Clean and Prepare
print("Preprocessing text...")
df_final['clean_text'] = df_final['text'].apply(preprocess_text)

# 3. Create Pipeline: TF-IDF + Logistic Regression
# We use ngram_range=(1, 3) to capture "great job breaking" patterns
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3), 
        max_features=10000,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        C=5.0, 
        class_weight='balanced', 
        max_iter=2000,
        solver='lbfgs'
    ))
])

# 4. Train
print("Fitting model...")
pipeline.fit(df_final['clean_text'], df_final['label'])

# 5. Save
MODEL_PATH = "sentiment_pipeline.pkl"
pickle.dump(pipeline, open(MODEL_PATH, "wb"))

print(f"DONE! {MODEL_PATH} has been updated with {len(df_final)} samples.")

# 6. Test on problematic cases
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
test_cases = [
    "Wow, great job breaking the app again.",
    "Zabardast app hai bhai, maza aa gaya.",
    "This update is totally useless.",
    "Dhanyawad support team."
]

print("\n--- Model Verification ---")
for tc in test_cases:
    pred = pipeline.predict([preprocess_text(tc)])[0]
    print(f"Text: {tc} => Prediction: {label_map[pred]}")
