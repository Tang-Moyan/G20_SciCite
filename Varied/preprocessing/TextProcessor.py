import re
import spacy


class TextPreprocessor:
    def __init__(self, model="en_core_web_sm", remove_stopwords=True):
        self.nlp = spacy.load(model, disable=['parser', 'ner'])
        self.remove_stopwords = remove_stopwords

    def remove_punctuation(self, text):
        """
        Remove punctuation from the text.

        Args:
        text (str): The input text.

        Returns:
        str: Text with punctuation removed.
        """
        # Preserve some punctuation marks like periods, exclamation marks
        preserved_punctuation = ".!?"
        return re.sub(rf'[^{preserved_punctuation}\w\s]', '', text)

    def to_lowercase(self, text):
        """
        Convert text to lowercase.

        Args:
        text (str): The input text.

        Returns:
        str: Lowercased text.
        """
        return text.lower()

    def tokenize(self, text):
        """
        Tokenize the text into words.

        Args:
        text (str): The input text.

        Returns:
        list: List of tokens (words).
        """
        return [token.text for token in self.nlp(text)]

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from the list of tokens.

        Args:
        tokens (list): List of tokens (words).

        Returns:
        list: List of tokens with stopwords removed.
        """
        stopwords = set(self.nlp.Defaults.stop_words)
        return [word for word in tokens if word not in stopwords]

    def preprocess_text(self, text):
        """
        Preprocess the input text by removing punctuation, converting to lowercase, tokenizing,
        and removing stopwords.

        Args:
        text (str): The input text.

        Returns:
        list: Preprocessed tokens.
        """
        text = self.remove_punctuation(text)
        text = self.to_lowercase(text)
        tokens = self.tokenize(text)
        if self.remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        return tokens


# Example usage
if __name__ == "__main__":
    text = "This is an example sentence, showing the process of text preprocessing!"

    preprocessor = TextPreprocessor(remove_stopwords=True)
    preprocessed_text = preprocessor.preprocess_text(text)
    print("Preprocessed text:", preprocessed_text)
