import os
import torch
import numpy as np
import pandas as pd
import pickle
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel


def glove_to_word2vec(glove_input_file, word2vec_output_file):
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_input_file, word2vec_output_file)


class PretrainedEmbeddingModel:
    def __init__(self, model_name='glove', file_path=None, test=False):
        self.model_name = model_name.lower()
        self.file_path = file_path
        self.test = test
        self.embeddings = None
        self.vocab_size = None
        self.vocab = {}
        self.embedding_matrix = None
        self.load_pretrained_embedding()

    def load_pretrained_embedding(self):
        if self.test:
            self.embedding_matrix, self.vocab = self.generate_random_embeddings()
        elif self.model_name == 'glove':
            self.embedding_matrix, self.vocab = self.load_glove_model()
        elif self.model_name == 'fasttext':
            self.embedding_matrix, self.vocab = self.load_fasttext_model()
        elif self.model_name == 'bert':
            self.embedding_matrix, self.vocab = self.load_bert_model()
        else:
            raise ValueError("Invalid model_name. Choose 'glove', 'fasttext', or 'bert'.")

    #---------------------- JUST USE THIS FUNCTION ----------------------#
    def load_glove_model(self):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.endswith('.npy'):
            # Special case if the embeddings are already in numpy format
            print("Loading saved numpy GloVe embeddings matrix...")
            embedding_matrix = np.load(self.file_path)
            self.vocab = pickle.load(open('glove_vocab.pkl', 'rb'))
            print("GloVe embeddings and vocab loaded.")
            return torch.from_numpy(embedding_matrix).float(), self.vocab
        
        if self.file_path.endswith('.word2vec.txt'):
            print("Loading Word2Vec format of GloVe embeddings...")
            glove_model = KeyedVectors.load_word2vec_format(self.file_path, binary=False)
            self.vocab = {word: idx for idx, word in enumerate(glove_model.index_to_key)}
            embedding_matrix = np.vstack([glove_model[word] for word in glove_model.index_to_key])

            # Save the vocab and embedding matrix for future use
            with open('glove_vocab.pkl', 'wb') as f:
                pickle.dump(self.vocab, f)
                print("GloVe vocab saved.")
            glove_embedding_matrix_name = self.file_path.replace('.txt', '.npy')
            np.save(glove_embedding_matrix_name, embedding_matrix)
            print("GloVe embeddings saved.")

            return torch.from_numpy(embedding_matrix).float(), self.vocab

        word2vec_output_file = self.file_path.replace('.txt', '.word2vec.txt')
        if not os.path.exists(word2vec_output_file):
            print("Converting GloVe to Word2Vec format...")
            glove_to_word2vec(self.file_path, word2vec_output_file)
        else:
            print("Existing Word2Vec format of GloVe embeddings found.")

        print("Loading GloVe embeddings...")

        glove_embedding_matrix_name = self.file_path.replace('.txt', '.npy')

        glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        self.vocab = {word: idx for idx, word in enumerate(glove_model.index_to_key)}
        embedding_matrix = np.vstack([glove_model[word] for word in glove_model.index_to_key])
        # Save the vocab and embedding matrix for future use
        with open('glove_vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
            print("GloVe vocab saved.")
        np.save(glove_embedding_matrix_name, embedding_matrix)
        
        print("GloVe embeddings saved.")

        return torch.from_numpy(embedding_matrix).float(), self.vocab

    def load_fasttext_model(self):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        fasttext_model = KeyedVectors.load_word2vec_format(self.file_path, binary=True)
        self.vocab = {word: idx for idx, word in enumerate(fasttext_model.index_to_key)}
        embedding_matrix = np.vstack([fasttext_model[word] for word in fasttext_model.index_to_key])
        print("FastText embeddings loaded.")
        return torch.from_numpy(embedding_matrix).float(), self.vocab

    def load_bert_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        embeddings = []
        self.vocab = {word: idx for idx, word in enumerate(tokenizer.vocab.keys())}
        print("Loading BERT embeddings...")
        for word in tokenizer.vocab.keys():
            idx = tokenizer.convert_tokens_to_ids(word)
            with torch.no_grad():
                output = model(torch.tensor([idx]).unsqueeze(0))
            embeddings.append(output.last_hidden_state.squeeze(0).mean(0).numpy())
        print("BERT embeddings loaded.")
        embedding_matrix = np.vstack(embeddings)
        return torch.from_numpy(embedding_matrix).float(), self.vocab

    def generate_random_embeddings(self):
        vocab_size = 10000  # Assuming a vocab size of 10000 for the random case
        embedding_dim = 300  # Assuming an embedding dimension of 300
        random_embeddings = np.random.rand(vocab_size, embedding_dim)
        vocab = {f'word{i}': i for i in range(vocab_size)}
        return torch.from_numpy(random_embeddings).float(), vocab

    def get_embeddings_and_vocab_size(self):
        if self.embedding_matrix is None or len(self.vocab) == 0:
            raise ValueError("Embeddings not loaded. Call load_pretrained_embedding first.")
        return self.embedding_matrix, self.vocab
    
    def get_vector_from_tokens(self, tokens, vector_length):
        if self.embedding_matrix is None or len(self.vocab) == 0:
            raise ValueError("Embeddings not loaded. Call load_pretrained_embedding first.")
        
        if len(tokens) > vector_length:
            tokens = tokens[:vector_length]
        elif len(tokens) < vector_length:
            tokens += ['<PAD>'] * (vector_length - len(tokens))

        return torch.tensor([self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens], dtype=torch.long)

