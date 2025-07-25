# Game Chat Toxicity Detector

This project is an end-to-end machine learning application designed to detect and flag toxic messages in real-time, similar to systems used in online gaming communities. It uses Natural Language Processing (NLP) to analyze text and provides a prediction on whether a message is toxic.

The entire application is containerized with Docker, making it portable, scalable, and easy to deploy.

---

## Features

* **Toxicity Classification:** Classifies input text as either "Toxic" or "Not Toxic".
* **Confidence Score:** Provides a confidence score for the toxicity prediction.
* **Simple Web Interface:** A clean, user-friendly UI to test the model in real-time.
* **REST API:** A backend API built with Flask to serve the model.
* **Containerized:** Fully containerized with Docker for easy setup and deployment.

---

## How It Works

The project is broken down into several phases:

1.  **Data Preprocessing:** A large dataset of over 150,000 labeled comments from Kaggle's [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) is cleaned and prepared. This involves lowercasing, removing punctuation and stopwords, and lemmatization.
2.  **Model Training:** A `Logistic Regression` model is trained on the processed data. The text is converted into numerical vectors using `TF-IDF` (Term Frequency-Inverse Document Frequency). The trained model and vectorizer are then saved.
3.  **API Development:** A Flask server (`app.py`) provides a `/predict` endpoint that takes a text message, preprocesses it in the same way as the training data, and returns a JSON response with the model's prediction.
4.  **Containerization:** The entire application is packaged into a Docker container using a `Dockerfile`. This creates a consistent and isolated environment, ensuring the application runs the same way everywhere.

<img width="657" height="466" alt="image" src="https://github.com/user-attachments/assets/968b3297-082b-4783-979b-63e182fa07b7" />

---

## Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas, NLTK
* **Frontend:** HTML, Tailwind CSS, JavaScript
* **Deployment:** Docker, Gunicorn

---

## How to Run Locally

There are two ways to run this project: using Docker (recommended) or setting up a local Python environment.

### Option 1: Running with Docker (Recommended)

This is the easiest and most reliable way to run the application.

**Prerequisites:**
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* The raw data file (`train.csv`) downloaded from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and placed in the root of the project folder.

**Instructions:**
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/toxicity-detector-api.git](https://github.com/your-username/toxicity-detector-api.git)
    cd toxicity-detector-api
    ```
2.  **Generate the Model Files:** The trained model files are not included in the repository. You must generate them first by running the training scripts.
    ```bash
    # First, set up a temporary Python environment to run the scripts
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

    # Run the data preparation and model training scripts
    python main_data_prep.py
    python train_model.py
    ```
3.  **Build the Docker image:**
    ```bash
    docker build -t toxicity-detector .
    ```
4.  **Run the Docker container:**
    ```bash
    docker run -p 5000:8000 toxicity-detector
    ```
5.  Open your web browser and navigate to `http://127.0.0.1:5000`.

### Option 2: Running with a Local Python Environment

**Prerequisites:**
* Python 3.9+
* `pip` and `venv`
* The raw data file (`train.csv`) downloaded from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and placed in the root of the project folder.

**Instructions:**
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/toxicity-detector-api.git](https://github.com/your-username/toxicity-detector-api.git)
    cd toxicity-detector-api
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare data and train the model:**
    ```bash
    python main_data_prep.py
    python train_model.py
    ```
5.  **Run the Flask application:**
    ```bash
    python app.py
    ```
6.  Open your web browser and navigate to `http://127.0.0.1:5000`.

---

## Project Structure

```
.
├── Raw Data/
│   └── train.csv.zip       # Original dataset from Kaggle
├── templates/
│   └── index.html          # Frontend HTML and JavaScript
├── .gitignore              # Files and folders to ignore by Git
├── Dockerfile              # Instructions to build the Docker image
├── app.py                  # Flask application with API endpoints
├── main_data_prep.py       # Script for initial data cleaning
├── requirements.txt        # Project dependencies
├── train_model.py          # Script to train the model
├── toxicity_model.joblib   # (Generated by train_model.py)
└── tfidf_vectorizer.joblib # (Generated by train_model.py)
