import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from base_line_model.BiLSTM_classifier import BiLSTMGRUClassifier
from myDataset import CustomDataset
from EDA import EDA
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from Embedding.load_pretrained_model import PretrainedEmbeddingModel
from sklearn.model_selection import train_test_split

print("READ ME:")
print("Ensure that you have downloaded the spacy en_core_web_sm model by running the following command in the terminal:")
print("python -m spacy download en_core_web_sm")
print("Ensure that you have the numpy embedding matrix file downloaded and saved into the Varied folder.")
input("Press any key once you are ready to train.")

# Initialize EDA and get data
eda = EDA()
preprocessed_strings, labels, label_confidence = eda.get_train(save_train=False)

# Create dataset and perform train/validation split
print("Creating dataset...")
dataset = preprocessed_strings
train_size = int(0.8 * len(labels))
val_size = len(labels) - train_size

print("Splitting dataset into training and validation sets...")
strings_train, strings_val, labels_train, labels_val = train_test_split(preprocessed_strings, labels, test_size=val_size)

# Model setup
print("Setting up model...")
model = LogisticRegression(multi_class='multinomial', max_iter=15)

# Training and Prediction
model.fit(strings_train, labels_train)
predicted = model.predict(strings_val)

# Calculate metrics
accuracy = accuracy_score(labels_val, predicted)
precision = precision_score(labels_val, predicted, average='macro')
recall = recall_score(labels_val, predicted, average='macro')
f1 = f1_score(labels_val, predicted, average='macro')

# Optionally, print final metrics for all recording points
print("Final Metrics:" + '\n' + 'accuracy: ' + accuracy + '\n' + 'precision: ' + precision + '\n' + 'recall: ' + recall + '\n' + 'f1_score: ' + f1)
