import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from base_line_model.BiLSTM_classifier import BiLSTMGRUClassifier
from myDataset import CustomDataset
from EDA import EDA
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from Embedding.load_pretrained_model import PretrainedEmbeddingModel

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize EDA and get data
eda = EDA()
preprocessed_strings, labels, label_confidence = eda.get_train(save_train=False)

# Load pretrained embeddings
embedding_model = PretrainedEmbeddingModel(model_name='glove', file_path='glove.42B.300d.txt', test=False)
embeddings_matrix, _ = embedding_model.get_embeddings_and_vocab_size()

# Create dataset and perform train/validation split
dataset = CustomDataset(preprocessed_strings, labels, label_confidence, model_name='glove',
                        file_path='glove.42B.300d.txt', use_label_smoothing=False, test=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print("Splitting dataset into training and validation sets...")
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set up DataLoader for both training and validation datasets
print("Loading data into DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
print("Finished loading data into DataLoader.")

# Model setup
print("Setting up model...")
embedding_dim = 300
vocab_size = len(embeddings_matrix)
model = BiLSTMGRUClassifier(input_dim=vocab_size, embedding_dim=embedding_dim, hidden_dim=256, output_dim=3,
                            pretrained_embeddings=embeddings_matrix)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training and validation loop
epochs = 2000
metrics = {}
print("Starting training loop...")
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1} Training"):
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU 上
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()

    # Validation and metrics recording every 50 epochs
    if (epoch + 1) % 50 == 0:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, leave=False, desc=f"Epoch {epoch + 1} Validation"):
                inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU 上
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Store metrics in a dictionary
        metrics[epoch + 1] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Save metrics to a JSON file
        with open(f'metrics_epoch_{epoch + 1}.json', 'w') as file:
            json.dump(metrics[epoch + 1], file, indent=4)

        # Save model weights
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        print(
            f"Epoch {epoch + 1}: Metrics saved. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Optionally, print final metrics for all recording points
print("Final Metrics:", json.dumps(metrics, indent=4))
