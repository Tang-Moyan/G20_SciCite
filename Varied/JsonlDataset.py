import torch
from torch.utils.data import Dataset
from preprocessing.TextProcessor import TextPreprocessor
from Embedding.load_pretrained_model import PretrainedEmbeddingModel
import json
import pandas as pd

class JsonlDataset(Dataset):
    '''
    Accepts a JSONL file and returns a PyTorch Dataset object.
    '''
    def __init__(self, jsonl_file_path,
                 embedding_model,
                 vector_length=300,
                ):
        
        self.jsonl_file_path = jsonl_file_path
        self.vector_length = vector_length
        self.text_preprocessor = TextPreprocessor(preserve_case=False)
        self.embedding_model = embedding_model

        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        parsed = []
        # For each item in the data, preprocess the text
        for item in data:
            text_tokens = self.text_preprocessor.preprocess_text(item.get('string', ''))
            vector = self.embedding_model.get_vector_from_tokens(text_tokens, self.vector_length)
            label = item.get('label', '')
            int_label = self.label_to_int(label)
            parsed.append({'vector': vector, 'label': int_label})

        return parsed


    def label_to_int(self, label):
        label_map = {'background': 0, 'result': 1, 'method': 2}
        return label_map.get(label, -1)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d['vector'], d['label']
