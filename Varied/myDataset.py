import torch
from torch.utils.data import Dataset
from preprocessing.TextProcessor import TextPreprocessor
from Embedding.load_pretrained_model import PretrainedEmbeddingModel


class CustomDataset(Dataset):
    def __init__(self, preprocessed_strings, labels, label_confidence, embedding_model=None, max_seq_length=300,
                 model_name='glove', file_path='glove_vector.txt', use_label_smoothing=False, embedding_dim=300, test=False):
        self.preprocessed_strings = preprocessed_strings
        self.labels = [self.label_to_int(label) for label in labels]
        self.label_confidence = label_confidence
        if use_label_smoothing:
            self.label_confidence = [0.9 if conf == 1 else 0.1 for conf in label_confidence]
        self.use_label_smoothing = use_label_smoothing
        if not test:
            if not embedding_model:
                self.embedding_model = PretrainedEmbeddingModel(model_name=model_name, file_path=file_path)
            else:
                self.embedding_model = embedding_model
        self.textProcessor = TextPreprocessor()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.test = test

    def __len__(self):
        return len(self.preprocessed_strings)

    def label_to_int(self, label):
        label_map = {'background': 0, 'result': 1, 'method': 2}
        return label_map.get(label, -1)

    def __getitem__(self, idx):
        text = self.preprocessed_strings[idx]
        words = self.textProcessor.tokenize(text)

        # 确保长度一致性
        # Adjust the length of the sequence
        if len(words) > self.max_seq_length:
            words = words[:self.max_seq_length]
        elif len(words) < self.max_seq_length:
            words += ['<PAD>'] * (self.max_seq_length - len(words))

        # 根据是否使用随机嵌入来获取索引
        # Get indices based on whether random embeddings are used
        if self.test:
            indices = torch.randint(high=10000, size=(self.max_seq_length,), dtype=torch.long)
        else:
            # Usually control flows here:
            indices = [self.embedding_model.vocab.get(word, self.embedding_model.vocab.get('<UNK>', 0)) for word in words]
            indices = torch.tensor(indices, dtype=torch.long)

        label = self.labels[idx]
        label_confidence = self.label_confidence[idx]
        if self.use_label_smoothing:
            num_classes = len(set(self.labels))
            smoothed_labels = torch.full((num_classes,), (1 - label_confidence) / (num_classes - 1), dtype=torch.float)
            smoothed_labels[label] = label_confidence
        else:
            smoothed_labels = torch.tensor([label], dtype=torch.long)

        return indices, smoothed_labels
