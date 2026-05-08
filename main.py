import joblib
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def main():
    print("--- Indian Mom Dialogue Classifier ---")
    print("Loading models...")
    
    try:
        emotion_model = joblib.load('emotion_model.pkl')
        threat_model = joblib.load('threat_model.pkl')
        ref_df = pd.read_csv('master_dataset.csv')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # To find hidden meaning, we need the TF-IDF vectors of the reference dialogues
    # The emotion_model pipeline has the vectorizer
    vectorizer = emotion_model.named_steps['tfidf']
    ref_vectors = vectorizer.transform(ref_df['clean_dialogue'].fillna(''))

    print("System Ready! Enter a dialogue (or 'quit' to exit).")
    
    while True:
        user_input = input("\nDialogue: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        clean_input = clean_text(user_input)
        if not clean_input:
            continue
        
        # 1. Predict Emotion
        emotion = emotion_model.predict([clean_input])[0]
        
        # 2. Predict Threat Level
        threat = threat_model.predict([clean_input])[0]
        
        # 3. Find Hidden Meaning using Cosine Similarity
        input_vector = vectorizer.transform([clean_input])
        similarities = cosine_similarity(input_vector, ref_vectors)
        best_match_idx = similarities.argmax()
        hidden_meaning = ref_df.iloc[best_match_idx]['hidden_meaning']
        
        # Output results
        print(f"-> Emotion: {emotion.upper()}")
        print(f"-> Hidden Meaning: \"{hidden_meaning}\"")
        
        # Color-coded threat (simplified for terminal)
        threat_display = threat.upper()
        if threat.lower() == 'rip':
            threat_display += " ⚠️⚠️⚠️"
        elif threat.lower() == 'high':
            threat_display += " ⚠️"
        
        print(f"-> Threat Level: {threat_display}")

if __name__ == "__main__":
    main()
