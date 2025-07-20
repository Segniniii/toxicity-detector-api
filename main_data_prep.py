# main_data_prep.py

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- First-time NLTK setup ---

# Run the script with these lines UNCOMMENTED once.
# After it succeeds, you can comment them out again.
#print("--- Downloading NLTK data (first-time setup) ---")
#nltk.download('punkt')
# Adding the missing 'punkt_tab' resource
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4') # Open Multilingual Wordnet
#print("--- NLTK data download complete ---")

# --- 2. Load and Explore the Data ---

def load_and_explore_data(filepath):
    """
    Loads the dataset, displays basic info, and creates a binary 'is_toxic' label.
    """
    print("--- Loading and Exploring Data ---")
    
    # Load the training data from the CSV file
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please make sure you have downloaded the data and placed it in the correct directory.")
        return None

    print("Dataset loaded successfully. Here are the first 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()

    # Define the columns that indicate toxicity
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create a single binary 'is_toxic' column
    # If any of the toxic_cols is 1, then is_toxic will be 1, otherwise 0.
    df['is_toxic'] = df[toxic_cols].max(axis=1)

    print("\nDistribution of the new 'is_toxic' label:")
    print(df['is_toxic'].value_counts(normalize=True))
    
    # We only need the comment text and our new label for the next phase
    df_processed = df[['comment_text', 'is_toxic']].copy()
    
    print("\nCreated a simplified DataFrame with 'comment_text' and 'is_toxic'.")
    return df_processed


# --- 3. Preprocess the Text Data ---

def preprocess_text(text):
    """
    Cleans and prepares a single text comment for modeling.
    """
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove special characters, numbers, and extra whitespace
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenization (split text into words)
    tokens = nltk.word_tokenize(text)
    
    # 4. Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    # 5. Join tokens back into a single string
    return " ".join(cleaned_tokens)


# --- Main Execution ---
if __name__ == '__main__':
    # --- Set the file path to your data ---
    # The 'r' before the string is CRITICAL. It makes it a "raw string" and
    # prevents the SyntaxError you were seeing.
    DATA_FILEPATH = r'C:\Users\ViG-CSAgent\Desktop\TOXIC CHAT\train.csv'
    
    df = load_and_explore_data(DATA_FILEPATH)
    
    if df is not None:
        print("\n--- Preprocessing Text Data ---")
        print("This will process the entire dataset and may take several minutes...")
        
        # Apply the preprocessing function to the 'comment_text' column
        df['clean_comment'] = df['comment_text'].apply(preprocess_text)
        
        # Fill any potential empty rows in 'clean_comment' that might result from preprocessing
        df['clean_comment'].fillna('', inplace=True)

        print("\nPreprocessing complete. Here's a sample of the original vs. cleaned comments:")
        print(df[['comment_text', 'clean_comment', 'is_toxic']].head())

        # Save the processed data to a new file to use in the next phase
        processed_filepath = 'processed_toxic_comments.csv'
        df.to_csv(processed_filepath, index=False)
        print(f"\nProcessed data saved to '{processed_filepath}'")
