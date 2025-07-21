import re
from sklearn.base import BaseEstimator, TransformerMixin

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
        if hasattr(X, 'apply'):
            return X.apply(self.preprocess)
        else:
            return [self.preprocess(x) for x in X] 