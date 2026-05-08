import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def train_models():
    # Load dataset
    df = pd.read_csv('master_dataset.csv')
    
    # Text and Labels
    X = df['clean_dialogue'].fillna('')
    y_emotion = df['emotion']
    y_threat = df['threat_level']
    
    # Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    # 1. Emotion Model
    emotion_model = Pipeline([
        ('tfidf', vectorizer),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    emotion_model.fit(X, y_emotion)
    print("Emotion model trained.")
    
    # 2. Threat Level Model
    threat_model = Pipeline([
        ('tfidf', vectorizer),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    threat_model.fit(X, y_threat)
    print("Threat level model trained.")
    
    # Save models and vectorizer
    joblib.dump(emotion_model, 'emotion_model.pkl')
    joblib.dump(threat_model, 'threat_model.pkl')
    
    # Also save the original dataframe for hidden meaning retrieval
    # We only need the cleaned dialogue and hidden meaning
    df[['clean_dialogue', 'hidden_meaning']].to_csv('reference_data.csv', index=False)
    print("Models and reference data saved.")

if __name__ == "__main__":
    train_models()
