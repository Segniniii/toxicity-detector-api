# app.py # this file is the backend API created using FLASK
# this file is the backend API created using FLASK
# uses incoming messages and "guesses" wether is a toxic comment.


# --- 1. Import Necessary Libraries ---
from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- 2. Define Helper Functions ---

def preprocess_text(text):
    """
    Cleans and prepares a single text comment for the model.
    This must be identical to the function used during training.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(cleaned_tokens)

# --- 3. Initialize App and Load Models ---
app = Flask(__name__)

# The NLTK download is now handled by the Dockerfile, so we don't need it here.

# Load the Model and Vectorizer once on startup
print("--- Loading model and vectorizer ---")
try:
    model = joblib.load('toxicity_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: model or vectorizer files not found. Please run train_model.py first.")
    model = None
    vectorizer = None

# --- 4. Define API Endpoints ---
@app.route('/')
def home():
    """
    Renders the main HTML page for the user interface.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives text input, predicts toxicity, and returns the result as JSON.
    """
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded, cannot make a prediction.'}), 500

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid input. "message" key is required.'}), 400
    
    message = data['message']
    
    # Process and predict
    cleaned_message = preprocess_text(message)
    message_tfidf = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_tfidf)
    prediction_proba = model.predict_proba(message_tfidf)

    # Format the response
    is_toxic = bool(prediction[0])
    confidence = float(prediction_proba[0][1])
    
    return jsonify({
        'is_toxic': is_toxic,
        'confidence': f"{confidence:.2f}"
    })
