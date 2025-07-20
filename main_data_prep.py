# main_data_prep.py

# --- 1. Import Libraries ---
# Used for data manipulation and reading CSV files.
import pandas as pd 
# Although not used directly, pandas depends on numpy.
import numpy as np 
# Essential for text pattern matching and removal (e.g., punctuation).
import re 
# The Natural Language Toolkit, our primary library for NLP tasks.
import nltk 
# For accessing lists of common words (stopwords) to filter out.
from nltk.corpus import stopwords 
# For reducing words to their base/dictionary form (lemmatization).
from nltk.stem import WordNetLemmatizer 

# --- Note on NLTK Data ---
# The first time you run any NLTK process, it may need to download necessary
# data packages. If you encounter a LookupError, you can run the following
# lines in a Python interactive session to download them manually:
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')

# --- 2. Data Loading and Initial Transformation ---

def load_and_explore_data(filepath):
    """
    Loads the raw dataset from a CSV file, creates a simplified binary 'is_toxic' 
    label, and provides an initial overview of the data.
    """
    print("--- Phase 1: Loading and Exploring Data ---")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure you have downloaded the data from Kaggle and it's in the correct directory.")
        return None

    print("Dataset loaded successfully. Displaying first 5 rows:")
    print(df.head())
    print("\nDataset Information:")
    df.info()

    # Define the columns that represent various forms of toxicity.
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create a single 'is_toxic' column. A comment is marked as toxic (1) 
    # if it's flagged in any of the toxic categories, otherwise it's non-toxic (0).
    df['is_toxic'] = df[toxic_cols].max(axis=1)

    print("\nDistribution of 'is_toxic' vs. 'not_toxic' comments:")
    # `normalize=True` shows the distribution as a percentage.
    print(df['is_toxic'].value_counts(normalize=True))
    
    # For our model, we only need the comment text and our new binary label.
    # We create a copy to avoid SettingWithCopyWarning in pandas.
    df_processed = df[['comment_text', 'is_toxic']].copy()
    
    print("\nCreated a simplified DataFrame for preprocessing.")
    return df_processed


# --- 3. Text Preprocessing Function ---

def preprocess_text(text):
    """
    Applies a series of cleaning steps to a single string of text to prepare
    it for machine learning.
    """
    # Initialize NLTK tools.
    lemmatizer = WordNetLemmatizer()
    # Using a set provides a significant speed-up for checking stopwords.
    stop_words = set(stopwords.words('english'))

    # Step 1: Convert all text to lowercase.
    text = text.lower()
    
    # Step 2: Use regular expressions to remove anything that is not a letter or space.
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Step 3: Split the text into a list of individual words (tokens).
    tokens = nltk.word_tokenize(text)
    
    # Step 4: Remove stopwords and apply lemmatization to each word.
    # This list comprehension is an efficient way to build the cleaned list.
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    # Step 5: Join the cleaned tokens back into a single string.
    return " ".join(cleaned_tokens)


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define the path to the raw data file.
    # Using a raw string (r'...') is crucial on Windows to prevent path errors.
    DATA_FILEPATH = r'train.csv'
    
    # Load and perform initial transformation on the data.
    df = load_and_explore_data(DATA_FILEPATH)
    
    # Proceed only if the DataFrame was loaded successfully.
    if df is not None:
        print("\n--- Phase 2: Preprocessing Text Data ---")
        print("This will process the entire dataset and may take several minutes...")
        
        # Apply our cleaning function to every comment in the 'comment_text' column.
        df['clean_comment'] = df['comment_text'].apply(preprocess_text)
        
        # A safety check: if a comment becomes empty after cleaning, fill it with an empty string.
        df['clean_comment'].fillna('', inplace=True)

        print("\nPreprocessing complete. Here's a sample of the results:")
        print(df[['comment_text', 'clean_comment', 'is_toxic']].head())

        # Save the fully processed data to a new CSV file for the next phase (model training).
        processed_filepath = 'processed_toxic_comments.csv'
        df.to_csv(processed_filepath, index=False)
        print(f"\nProcessed data saved to '{processed_filepath}'")

