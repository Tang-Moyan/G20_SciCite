import os
import torch
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel


class PretrainedEmbeddingModel:
    def __init__(self, model_name='glove', file_path=None):
        """
        Initialize the PretrainedEmbeddingModel class with the specified pretrained model.

        Args:
        model_name (str): Name of the pretrained embedding model to load. Default is 'glove'.
        file_path (str): File path to the pretrained embedding model if applicable.
        """
        self.model_name = model_name.lower()  # Normalize model name to lowercase
        self.file_path = file_path
        self.embeddings = None
        self.load_pretrained_embedding()

    def load_pretrained_embedding(self):
        """
        Load the pretrained embedding model specified by the model_name parameter.

        """
        if self.model_name == 'glove':
            self.embeddings = self.load_glove_model()
        elif self.model_name == 'fasttext':
            self.embeddings = self.load_fasttext_model()
        elif self.model_name == 'bert':
            self.embeddings = self.load_bert_model()
        else:
            raise ValueError("Invalid model_name. Please choose from 'glove', 'fasttext', or 'bert'.")

    def load_glove_model(self):
        """
        Load the GloVe model from the specified file path and get embeddings.

        Returns:
        glove_embeddings (dict): Embeddings from the loaded GloVe model.
        """
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        glove_model = KeyedVectors.load_word2vec_format(self.file_path, binary=False)
        glove_embeddings = {word: glove_model[word] for word in glove_model.index2word}
        return glove_embeddings

    def load_fasttext_model(self):
        """
        Load the FastText model from the specified file path and get embeddings.

        Returns:
        fasttext_embeddings (dict): Embeddings from the loaded FastText model.
        """
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        fasttext_model = KeyedVectors.load(self.file_path)
        fasttext_embeddings = {word: fasttext_model[word] for word in fasttext_model.index2word}
        return fasttext_embeddings

    def load_bert_model(self):
        """
        Load the BERT model and get embeddings.

        Returns:
        bert_embeddings (torch.Tensor): Embeddings from the loaded BERT model.
        """
        model = BertModel.from_pretrained(self.model_name)
        tokenizer = BertTokenizer.from_pretrained(self.model_name)

        return model, tokenizer

    def get_embedding(self, word):
        """
        Get the embedding for the specified word.

        Args:
        word (str): The word to get embedding for.

        Returns:
        embedding (torch.Tensor or numpy.ndarray): The embedding for the word.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Please call load_pretrained_embedding first.")

        if self.model_name in ['glove', 'fasttext']:
            return self.embeddings.get(word)
        elif self.model_name == 'bert':
            model, tokenizer = self.embeddings
            input_text = word  # Assuming single word is provided
            inputs = tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.squeeze(0)
            return embedding
        else:
            raise ValueError("Invalid model_name. Please choose from 'glove', 'fasttext', or 'bert'.")


# Example usage
if __name__ == "__main__":
    # Using GloVe model (default)
    glove_model_path = 'path/to/glove_vectors.txt'
    embedding_model = PretrainedEmbeddingModel(model_name='glove', file_path=glove_model_path)
    print("GloVe embeddings loaded successfully")
    word = 'example'
    print(f"Embedding for '{word}':", embedding_model.get_embedding(word))

    # Using FastText model
    fasttext_model_path = 'path/to/fasttext_vectors.bin'
    embedding_model = PretrainedEmbeddingModel(model_name='fasttext', file_path=fasttext_model_path)
    print("FastText embeddings loaded successfully")
    word = 'example'
    print(f"Embedding for '{word}':", embedding_model.get_embedding(word))

    # Using BERT model
    bert_model_name = 'bert-base-uncased'
    embedding_model = PretrainedEmbeddingModel(model_name='bert', file_path=bert_model_name)
    print("BERT embeddings loaded successfully")
    word = 'example'
    print(f"Embedding for '{word}':", embedding_model.get_embedding(word))
