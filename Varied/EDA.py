from preprocessing.TextProcessor import TextPreprocessor
from dataset_reader import jsonl_file_reader
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the data from the JSONL file
file_path = '../data/train.jsonl'
reader = jsonl_file_reader.JSONLReader(file_path)
data_list = reader.read_file()
# Initialize the text processor
strings = [items.get('string', None) for items in data_list]
labels = [items.get('label', None) for items in data_list]

# Fill NA with label confidence with mean value
label_confidence = [items.get('label_confidence', None) for items in data_list]
na_counts = label_confidence.count(None)
print(na_counts)
mean_confidence = sum([conf for conf in label_confidence if conf is not None]) / (len(label_confidence) - na_counts)
print(mean_confidence)
label_confidence = [mean_confidence if conf is None else conf for conf in label_confidence]
# Count the number of label_confidence below 0.5
low_confidence = sum([1 for conf in label_confidence if conf < 0.5])
# print(low_confidence)
# Fine the text of label_confidence below 0.5
for i, conf in enumerate(label_confidence):
    if conf < 0.5:
        print(data_list[i])
        print(data_list[i]['label_confidence'])
print("Strings:", strings[:5])
print("Labels:", labels[:5])
print("Label Confidence:", label_confidence[:5])
print("Labels counts:", {label: labels.count(label) for label in set(labels)})

# Generate a distribution for label confidence
plt.hist(label_confidence, bins=20, edgecolor='black')
plt.title('Distribution of Label Confidence')
plt.xlabel('Confidence Level')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Preprocess the text data
processor = TextPreprocessor(remove_stopwords=True)
preprocessed_strings = [processor.preprocess_text(string) for string in strings]
print("Preprocessed Strings:", preprocessed_strings[:5])