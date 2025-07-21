# Imports Flask for web serving, pickle for loading the model, os for file paths, re for regex, and scikit-learn/nltk for text processing.
from flask import Flask, render_template, request, session
import pickle
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import io
import base64
from markupsafe import Markup, escape
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Ensure VADER lexicon is available at server start
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# IMPORTANT: TextCleaner must be defined in the global scope before loading the pickled model.
class TextCleaner(BaseEstimator, TransformerMixin):
    def remove_html(self, text):
        return re.sub('<[^>]*>', '', text)

    def remove_non_words(self, text):
        return re.sub(r'[\W]+', ' ', text.lower())

    def extract_emojis(self, text):
        emojis = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        return ' '.join(emojis).replace('-', '')

    def preprocess(self, text):
        text = self.remove_html(text)
        text = self.remove_non_words(text)
        emojis = self.extract_emojis(text)
        return text + ' ' + emojis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accept both pandas Series and lists
        if hasattr(X, 'apply'):
            return X.apply(self.preprocess)
        else:
            return [self.preprocess(x) for x in X]



nltk.download('stopwords', quiet=True)
stop = stopwords.words('english')
porter = PorterStemmer()
#Defines a tokenizer that removes stopwords and applies stemming.
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word not in stop]

#Generates a word cloud image from text, encodes it as base64, and returns the string.
#Returns None if not enough words.
#Text: The text to generate a word cloud from.
#Stopwords_list: The list of stopwords to remove from the text.
#Width: The width of the word cloud image.
#Height: The height of the word cloud image.
def generate_wordcloud_base64(text, stopwords_list=None, width=400, height=200):
    """
    Generate a word cloud image from text, encode as base64, and return the string.
    Returns None if not enough words.
    """
    if not text or len(text.split()) < 3:
        return None
    stopwords_set = set(stopwords_list) if stopwords_list else set()
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        stopwords=stopwords_set,
        collocations=False
    )
    wc.generate(text)
    img_io = io.BytesIO()
    wc.to_image().save(img_io, format='PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    return img_base64

#Encapsulates model loading and prediction logic.
#Model Loading: Loads the model from a file using pickle.
#Prediction: Accepts a single text input and returns the predicted sentiment.
class SentimentModel:
    """Handles loading and prediction for the sentiment analysis model."""
    def __init__(self, model_path):
        # Ensure TextCleaner is defined/imported before loading the model to avoid AttributeError during pickle.load()
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, text):
        # Assumes the model has a predict method that takes a list of texts
        return self.model.predict([text])[0]

#Predicts the sentiment of the user's input text and returns the predicted label and its confidence score as a percentage.
#Confidence Score: The confidence score is a percentage that indicates how confident the model is in its prediction.
#It is calculated by the model's predict_proba method, which returns the probability of the predicted label.
#The confidence score is used to display the confidence score in the UI.
    def predict_with_confidence(self, text):
        """Returns the predicted label and its confidence score as a percentage."""
        label = self.model.predict([text])[0]
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([text])[0]
            # Get the index of the predicted label
            if hasattr(self.model, 'classes_'):
                idx = list(self.model.classes_).index(label)
            else:
                idx = 1 if label == 1 else 0
            confidence = proba[idx]
        else:
            confidence = 1.0  # fallback if predict_proba not available
        return label, round(confidence * 100, 2)

def highlight_sentiment_words(text):
    """
    Highlights positive and negative words in the input text using VADER lexicon.
    Returns HTML-safe string with <span> tags for sentiment words.
    """
    sia = SentimentIntensityAnalyzer()
    import re
    tokens = re.findall(r"\w+|[\s.,!?;]", text)
    highlighted = []
    for token in tokens:
        word = token.strip().lower()
        if not word or not word.isalpha():
            highlighted.append(escape(token))
            continue
        score = sia.polarity_scores(word)['compound']
        if score >= 0.5:
            highlighted.append(f'<span class="positive-word">{escape(token)}</span>')
        elif score <= -0.5:
            highlighted.append(f'<span class="negative-word">{escape(token)}</span>')
        else:
            highlighted.append(escape(token))
    return Markup(''.join(highlighted))

# Helper function to manage session-based prediction history
from functools import wraps

PREDICTION_HISTORY_KEY = 'prediction_history'
MAX_HISTORY = 5

def add_prediction_to_history(user_text, label, confidence):
    """
    Adds a prediction entry to the session history, keeping only the most recent MAX_HISTORY entries.
    Each entry is a dict: {input, label, confidence}
    Ensures all values are JSON serializable (e.g., confidence is float/int, not numpy.int64).
    """
    # Ensure confidence is a native Python float (not numpy type)
    try:
        confidence_py = float(confidence)
    except Exception:
        confidence_py = confidence
    entry = {
        'input': user_text,
        'label': str(label),
        'confidence': confidence_py
    }
    history = session.get(PREDICTION_HISTORY_KEY, [])
    history.append(entry)
    # Keep only the last MAX_HISTORY entries
    history = history[-MAX_HISTORY:]
    session[PREDICTION_HISTORY_KEY] = history
    session.modified = True
    return history

def get_prediction_history():
    """Returns the current session's prediction history (list of dicts)."""
    return session.get(PREDICTION_HISTORY_KEY, [])

#Defines a Flask app wrapper for sentiment analysis.
#App Initialization: Creates a Flask app instance and loads the sentiment model.
#Route Setup: Sets up a route for the home page that handles GET and POST requests.
#Prediction: Predicts the sentiment of the user's input text.
class SentimentApp:
    """Flask app wrapper for sentiment analysis."""
    def __init__(self, model_path):
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')  # For session security
        self.model = SentimentModel(model_path)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            prediction = None
            confidence = None
            user_text = ''
            wordcloud_img = None
            wordcloud_fallback = None
            highlighted_text = None
            prediction_history = get_prediction_history()
            def label_to_text(label):
                if str(label).lower() in ['1', 'pos', 'positive']:
                    return 'Positive'
                elif str(label).lower() in ['0', 'neg', 'negative']:
                    return 'Negative'
                else:
                    return 'Neutral'
            if request.method == 'POST':
                user_text = request.form.get('user_text', '')
                if user_text:
                    prediction_raw, confidence = self.model.predict_with_confidence(user_text)
                    prediction = label_to_text(prediction_raw)
                    cleaner = TextCleaner()
                    cleaned = cleaner.preprocess(user_text)
                    cleaned_words = [w for w in cleaned.split() if w not in stop]
                    cleaned_text = ' '.join(cleaned_words)
                    wordcloud_img = generate_wordcloud_base64(cleaned_text, stop)
                    if not wordcloud_img:
                        wordcloud_fallback = 'Not enough words to generate a word cloud.'
                    highlighted_text = highlight_sentiment_words(user_text)
                    # Add to session-based prediction history with descriptive label
                    prediction_history = add_prediction_to_history(user_text, prediction, confidence)
            # Map history labels to descriptive text for display
            for entry in prediction_history:
                entry['label'] = label_to_text(entry['label'])
            return render_template('index.html', prediction=prediction, confidence=confidence, user_text=user_text, wordcloud_img=wordcloud_img, wordcloud_fallback=wordcloud_fallback, highlighted_text=highlighted_text, prediction_history=prediction_history)

        @self.app.route('/clear_history', methods=['POST'])
        def clear_history():
            session.pop(PREDICTION_HISTORY_KEY, None)
            return self.app.redirect('/')

    def run(self, **kwargs):
        self.app.run(**kwargs)

#Main execution block that initializes the app and runs it.
#Model Path: Specifies the path to the sentiment model file.
#App Initialization: Creates an instance of SentimentApp with the model path.
#App Run: Runs the Flask app in debug mode.
# For Gunicorn: expose the Flask app object at the top level
model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
sentiment_app = SentimentApp(model_path)
app = sentiment_app.app

if __name__ == '__main__':
    # The original __main__ block is now handled by Gunicorn, so this will not be executed directly.
    # If you want to run it locally for testing, you would need to set up a WSGI server.
    # For now, we'll keep it as is, but it won't be the entry point for the app.
    # The app object is now globally accessible as 'app'.
    pass 