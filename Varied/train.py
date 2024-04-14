import time
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from base_line_model.BiLSTM_classifier import BiLSTMGRUClassifier
from JsonlDataset import JsonlDataset
from EDA import EDA
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from Embedding.load_pretrained_model import PretrainedEmbeddingModel
from torch import nn

MODEL_NAME = "BiLSTMGRUClassifier.pth"

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 设置随机种子以确保结果可以复现
torch.manual_seed(42)

# Initialize EDA and get data
eda = EDA()
preprocessed_strings, labels, label_confidence = eda.get_train(save_train=False)

# Load pretrained embeddings
embedding_model = PretrainedEmbeddingModel(model_name='glove', file_path='glove.42B.300d.npy', test=False)
embeddings_matrix, _ = embedding_model.get_embeddings_and_vocab_size()

# Create dataset and perform train/validation/test split
dataset = JsonlDataset(jsonl_file_path='../data/train.jsonl', embedding_model=embedding_model)
train_size = int(0.7 * len(dataset))
test_size = int(0.15 * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                        generator=torch.Generator().manual_seed(42))

# Set up DataLoader for training, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model setup
model = BiLSTMGRUClassifier(input_dim=len(embeddings_matrix), embedding_dim=300, hidden_dim=256, output_dim=3,
                            pretrained_embeddings=embeddings_matrix)
model.to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training and validation loop
epochs = 200
metrics = {}
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    total_loss = 0
    num_batches = 0
    for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1} Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1} Loss: {avg_loss}")

    if (epoch + 1) % 5 == 0:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Validation Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        # Save metrics and model state
        metrics[epoch + 1] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
                              'loss': avg_loss}
        torch.save(model.state_dict(), f"{MODEL_NAME}_epoch_{epoch + 1}.pth")

# Save final metrics to a JSON file
with open('metrics.json', 'w') as file:
    json.dump(metrics, file, indent=4)

print("Training completed.")

# # Save metrics to a JSON file
# with open(f'metrics.json', 'w') as file:
#     json.dump(metrics, file, indent=4)
#
# # Save model weights
# torch.save(model.state_dict(), MODEL_NAME)
# print("Metrics saved. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
#
# # -----------------------------------------------------------
# exit()
# # Do a test load and prediction
# print("Loading model for testing...")
# model = BiLSTMGRUClassifier(input_dim=vocab_size, embedding_dim=embedding_dim, hidden_dim=256, output_dim=3,
#                             pretrained_embeddings=embeddings_matrix)
# model.load_state_dict(torch.load(f'BiLSTMGRUClassifier.pth'))
# model.to(device)
#
# # Predict a test sentence (use model.predict() for a single sentence)
# test_sentence = "The paper is about the COVID-19 pandemic."
