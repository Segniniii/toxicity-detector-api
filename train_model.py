# train_model.py

# --- 1. Import Necessary Libraries ---
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving the model and vectorizer

# --- 2. Load the Processed Data ---
def load_data(filepath):
    """
    Loads the preprocessed data from the specified CSV file.
    """
    print("--- Loading Processed Data ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please make sure you have the 'processed_toxic_comments.csv' file from Phase 1.")
        return None
    
    # Handle cases where 'clean_comment' might be empty (NaN) after loading
    # Updated to avoid the FutureWarning
    df['clean_comment'] = df['clean_comment'].fillna('')
    
    print("Data loaded successfully.")
    return df

# --- 3. Train the Model ---
def train_and_evaluate(df):
    """
    Vectorizes the text, splits the data, trains a model, and evaluates it.
    """
    if df is None:
        return

    print("\n--- Starting Model Training and Evaluation ---")
    
    # Define our features (X) and target (y)
    X = df['clean_comment']
    y = df['is_toxic']
    
    # --- Vectorization using TF-IDF ---
    print("Vectorizing text data with TF-IDF...")
    # We are creating a TF-IDF Vectorizer. max_features=5000 means we'll only
    # consider the top 5000 most frequent words to keep our model manageable.
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # --- Splitting Data into Training and Testing sets ---
    # We'll use 80% of the data for training and 20% for testing.
    # random_state=42 ensures we get the same split every time we run the code.
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- Training the Logistic Regression Model ---
    print("Training the Logistic Regression model...")
    model = LogisticRegression(max_iter=1000) # max_iter helps the model converge
    model.fit(X_train, y_train)
    
    # --- Evaluating the Model ---
    print("Evaluating the model on the test set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    # A confusion matrix shows us True Positives, True Negatives,
    # False Positives, and False Negatives.
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    # This report gives us precision, recall, and f1-score for each class.
    print(classification_report(y_test, y_pred, target_names=['Not Toxic (0)', 'Toxic (1)']))

    # --- 4. Save the Model and Vectorizer for Future Use ---
    print("\n--- Saving the model and vectorizer ---")
    joblib.dump(model, 'toxicity_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model saved as 'toxicity_model.joblib'")
    print("Vectorizer saved as 'tfidf_vectorizer.joblib'")


# --- Main Execution ---
if __name__ == '__main__':
    # The file created in the previous phase
    PROCESSED_DATA_FILEPATH = 'processed_toxic_comments.csv'
    
    dataframe = load_data(PROCESSED_DATA_FILEPATH)
    train_and_evaluate(dataframe)

