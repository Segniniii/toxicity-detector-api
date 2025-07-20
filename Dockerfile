# Stage 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Stage 2: Set the working directory in the container
WORKDIR /app

# Stage 3: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Download NLTK data during the build process
# This is more efficient than downloading every time the container starts.
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords wordnet omw-1.4 punkt_tab

# Stage 5: Copy the rest of the application's code into the container
COPY . .

# Stage 6: Make port 8000 available to the world outside this container
EXPOSE 8000

# Stage 7: Define the command to run the app using gunicorn
CMD ["/usr/local/bin/gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
