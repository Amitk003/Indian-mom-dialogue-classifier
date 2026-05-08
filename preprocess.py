import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def prepare_data():
    # Load datasets
    df1 = pd.read_csv('indian_mom_dialogue_classifier_dataset.csv')
    df2 = pd.read_csv('indian_mom_dialogue_dataset.csv')

    # Merge datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Clean dialogue text
    df['clean_dialogue'] = df['dialogue'].apply(clean_text)

    # Standardize labels
    df['emotion'] = df['emotion'].str.lower().str.strip()
    df['threat_level'] = df['threat_level'].str.lower().str.strip()

    # Drop duplicates
    df = df.drop_duplicates(subset=['clean_dialogue'])

    print(f"Total dialogues after merging: {len(df)}")
    print("\nEmotion Distribution:")
    print(df['emotion'].value_counts())
    
    print("\nThreat Level Distribution:")
    print(df['threat_level'].value_counts())

    # Save master dataset
    df.to_csv('master_dataset.csv', index=False)
    print("\nMaster dataset saved as 'master_dataset.csv'")

if __name__ == "__main__":
    prepare_data()
