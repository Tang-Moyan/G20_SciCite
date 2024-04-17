import re
import spacy

# Download the en_core_web_sm model by running the following command in the terminal:
# python -m spacy download en_core_web_sm


class TextPreprocessor:
    def __init__(self,
                 model="en_core_web_sm",
                 remove_stopwords=False,
                 preserve_punctuation=True,
                 preserve_case=True
                ):
        self.nlp = spacy.load(model, disable=['parser', 'ner'])
        self.remove_stopwords_flag = remove_stopwords
        self.vocab = set()  # Initialize an empty set to store vocabulary
        self.preserve_punctuation = preserve_punctuation

        self.preserve_case = preserve_case

    def remove_punctuation(self, text):
        preserved_punctuation = ".!?"
        return re.sub(rf'[^{preserved_punctuation}\w\s]', '', text)

    def to_lowercase(self, text):
        return text.lower()

    def tokenize(self, text):
        return [token.text for token in self.nlp(text)]

    def remove_stopwords(self, tokens):
        stopwords = set(self.nlp.Defaults.stop_words)
        return [word for word in tokens if word not in stopwords]

    def update_vocab(self, tokens):
        # This method updates the vocabulary set with new tokens
        self.vocab.update(tokens)

    def get_vocab_size(self):
        # This method returns the size of the vocabulary
        return len(self.vocab)

    def preprocess_text(self, text):
        if not self.preserve_punctuation:
            text = self.remove_punctuation(text)

        if not self.preserve_case:
            text = self.to_lowercase(text)

        tokens = self.tokenize(text)

        if self.remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)

        self.update_vocab(tokens)  # Update the vocabulary with tokens from the processed text
        return tokens


# Example usage
if __name__ == "__main__":
    text = "It was observed that frataxin interacts with the iron-sulfur cluster assembly complex, forming a complex that is essential for the biogenesis of iron-sulfur clusters."

    preprocessor = TextPreprocessor()
    preprocessed_text = preprocessor.preprocess_text(text)
    print("Preprocessed text:", preprocessed_text)
    print("Vocabulary size:", preprocessor.get_vocab_size())
