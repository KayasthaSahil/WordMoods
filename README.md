# Sentiment Analysis Web Application

This is a Flask-based web application that performs sentiment analysis on user-provided text. It uses a pre-trained machine learning model to classify text as either positive or negative. The application also provides additional text analysis features like word cloud generation and sentiment-aware word highlighting.

## Features

- **Sentiment Prediction**: Classifies input text into 'Positive' or 'Negative' categories.
- **Confidence Score**: Displays the model's confidence in its prediction.
- **Word Cloud**: Generates a word cloud from the input text to visualize word frequency.
- **Sentiment Highlighting**: Highlights positive and negative words in the input text.
- **Prediction History**: Keeps a history of the last 5 predictions.
- **Responsive UI**: A clean and simple user interface that works on different devices.

## How It Works

The application uses a `Logistic Regression` model trained on a movie review dataset. The text processing pipeline involves:
1.  **Text Cleaning**: Removing HTML tags, and non-word characters.
2.  **Tokenization and Stemming**: The text is tokenized, and words are stemmed to their root form. Stopwords are also removed.
3.  **Vectorization**: The cleaned text is converted into a numerical representation using `TfidfVectorizer`.

The front end is built with HTML, CSS, and a little bit of JavaScript for a better user experience.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    flask run
    ```
    Or using Gunicorn:
    ```bash
    gunicorn app:app
    ```
4.  Open your browser and navigate to `http://127.0.0.1:5000`.

## File Structure

-   `app.py`: The main Flask application file.
-   `sentiment_model.pkl`: The pre-trained sentiment analysis model.
-   `text_cleaner.py`: Contains the `TextCleaner` class for preprocessing text.
-   `templates/index.html`: The main HTML file for the user interface.
-   `static/style.css`: The CSS file for styling the application.
-   `static/uiux.js`: JavaScript for UI interactions.
-   `Procfile`: For deployment on services like Heroku.
-   `requirements.txt`: A list of Python packages required for the project.
