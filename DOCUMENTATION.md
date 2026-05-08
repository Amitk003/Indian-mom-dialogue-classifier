# Project Documentation

## Implementation Steps

### 1. Data Preparation
- Merged two CSV datasets: `indian_mom_dialogue_classifier_dataset.csv` and `indian_mom_dialogue_dataset.csv`.
- Cleaned the text by removing special characters and converting to lowercase.
- Handled missing values and standardized category labels.

### 2. Feature Engineering
- Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert Hinglish text into numerical vectors.
- Configured n-grams (1, 2) to capture specific phrases and word combinations.

### 3. Model Development
- Trained a Logistic Regression model for Emotion classification.
- Trained a separate classifier for Threat Level prediction.
- Developed a retrieval mechanism for Hidden Meaning based on text similarity.

### 4. Interface
- Developed a Terminal-based CLI (`main.py`) that takes user input, processes it through the trained models, and outputs the results directly to the console.
- Used Cosine Similarity for the retrieval-based "Hidden Meaning" translator.

## File Structure
- `master_dataset.csv`: The combined and cleaned dataset (Primary dataset).
- `main.py`: The main entry point for the terminal application.
- `train.py`: Generates machine learning models from the master dataset.
- `emotion_model.pkl` & `threat_model.pkl`: Serialized ML models.
- `preprocess.py`: Historical script used for initial data merging and cleaning.

## How to Run
1. Install dependencies: `pip install pandas scikit-learn joblib`
2. Prepare data: `python preprocess.py`
3. Train models: `python train.py`
4. Run application: `python main.py`

## Technical Notes
- Logistic Regression was chosen for its efficiency with smaller datasets and its ability to handle multi-class text classification effectively.
- The use of Hinglish required custom preprocessing as standard English stop-word lists are not applicable.
- Semantic search (Cosine Similarity) ensures that even if a dialogue isn't a perfect match, the system can still find a relevant "Hidden Meaning".
