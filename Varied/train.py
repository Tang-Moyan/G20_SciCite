import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from base_line_model.BiLSTM_classifier import BiLSTMGRUClassifier
from Embedding.load_pretrained_model import PretrainedEmbeddingModel
from sklearn.metrics import accuracy_score, precision_score

# Assuming some code exists to prepare your dataset
# dataset = YourCustomDataset()

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained word vectors
# This part will vary depending on how your PretrainedEmbeddingModel class is implemented
embedding_model = PretrainedEmbeddingModel(model_name='glove', file_path='path/to/your/glove/file')
# Ensure this returns the actual embedding matrix and vocab size
pretrained_embeddings, vocab_size = embedding_model.get_embeddings_and_vocab_size()

# Initialize the model
embedding_dim = pretrained_embeddings.shape[1]  # Dimension of pretrained word vectors
hidden_dim = 256  # Chosen hidden dimension size
output_dim = 2  # Number of output classes; adjust based on your specific task
model = BiLSTMGRUClassifier(input_dim=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Use pretrained embeddings in the model
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.requires_grad = False  # Optionally freeze the embeddings

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation loop
epochs = 100
for epoch in range(epochs):
    # Training loop
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation loop
    if (epoch + 1) % 50 == 0:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch + 1}: Accuracy: {acc}, Precision: {precision}")

        # Optionally save the model
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
