import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from base_line_model.BiLSTM_classifier import BiLSTMGRUClassifier
#from myDataset import CustomDataset
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

# Initialize EDA and get data
eda = EDA()
preprocessed_strings, labels, label_confidence = eda.get_train(save_train=False)

# Load pretrained embeddings
embedding_model = PretrainedEmbeddingModel(model_name='glove', file_path='glove.42B.300d.npy', test=False)
embeddings_matrix, _ = embedding_model.get_embeddings_and_vocab_size()

# Create dataset and perform train/validation split
print("Creating dataset...")
#dataset = CustomDataset(preprocessed_strings, labels, label_confidence, model_name='glove' embedding_model=embedding_model, use_label_smoothing=False, test=True)

# JsonlDataset is a class that consumes a .jsonl file (like the one given in Scicite) and also an embedding model like the one declared above and returns a PyTorch Dataset object.
# The dataset contains individual sample of (vector, label, label_confidence) for each item in the JSONL file. The vector is pytorch tensor of the embeddings of the text, label is the label of the text, and label_confidence is the confidence of the label.
dataset = JsonlDataset(jsonl_file_path='../data/train.jsonl', embedding_model=embedding_model)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print("Splitting dataset into training and validation sets...")
#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set up DataLoader for both training and validation datasets
print("Loading data into DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("Finished loading data into DataLoader.")

# Model setup
print("Setting up model...")
embedding_dim = 300
vocab_size = len(embeddings_matrix)
model = BiLSTMGRUClassifier(input_dim=vocab_size, embedding_dim=embedding_dim, hidden_dim=256, output_dim=3,
                            pretrained_embeddings=embeddings_matrix)
model.to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training and validation loop
epochs = 20
metrics = {}
print("Starting training loop...")
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    total_loss = 0
    num_batches = 0
    for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1} Training"):
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU 上
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / num_batches

    print(f"Epoch {epoch + 1} Loss: {avg_loss}\n")

    if (epoch + 1) % 5 == 0:
        print("Evaluating on validation set...")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, leave=False, desc=f"Epoch {epoch + 1} Validation"):
                inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU 上
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, 1)
                predicted = torch.argmax(probabilities, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        # Print a label and prediction for debugging
        print("All Labels:", all_labels)
        print("All Predictions:", all_preds)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1} Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Loss: {avg_loss}\n")

        metrics[epoch + 1] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print("Evaluting on training set...")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1} Training"):
                inputs, labels = inputs.to(device), labels.to(device)  # 将输入数据和标签移动到 GPU 上
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

        # Print a label and prediction for debugging
        print("All Labels:", all_labels)
        print("All Predictions:", all_preds)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1} Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Loss: {avg_loss}\n")

# Save metrics to a JSON file
with open(f'metrics.json', 'w') as file:
    json.dump(metrics, file, indent=4)

# Save model weights
torch.save(model.state_dict(), MODEL_NAME)
print("Metrics saved. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

#-----------------------------------------------------------
exit()
# Do a test load and prediction
print("Loading model for testing...")
model = BiLSTMGRUClassifier(input_dim=vocab_size, embedding_dim=embedding_dim, hidden_dim=256, output_dim=3,
                            pretrained_embeddings=embeddings_matrix)
model.load_state_dict(torch.load(f'BiLSTMGRUClassifier.pth'))
model.to(device)

# Predict a test sentence (use model.predict() for a single sentence)
test_sentence = "The paper is about the COVID-19 pandemic."
