import os
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.TextProcessor import TextPreprocessor
from dataset_reader import jsonl_file_reader


class EDA:
    def save_to_csv(self, texts, labels, confidences, file_path):
        # Create a DataFrame
        df = pd.DataFrame({
            "text": texts,
            "label": labels,
            "label_confidence": confidences
        })
        df.to_csv(file_path, index=False)

    def load_from_csv(self, file_path):
        # Load the data from the CSV file
        df = pd.read_csv(file_path)
        preprocessed_strings = df['text'].tolist()
        labels = df['label'].tolist()
        label_confidence = df['label_confidence'].tolist()
        return preprocessed_strings, labels, label_confidence

    def get_train(self, save_train=True):
        # Check if CSV file already exists
        csv_file_path = '../data/train.csv'
        if os.path.exists(csv_file_path):
            return self.load_from_csv(csv_file_path)
        else:
            # Load and preprocess the data from JSONL file
            preprocessed_strings, labels, label_confidence = self.load_and_preprocess_data()

            # Save the preprocessed data to a CSV file
            if save_train:
                self.save_to_csv(preprocessed_strings, labels, label_confidence, csv_file_path)

            return preprocessed_strings, labels, label_confidence

    def load_and_preprocess_data(self):
        # Load the data from the JSONL file
        file_path = '../data/train.jsonl'
        reader = jsonl_file_reader.JSONLReader(file_path)
        data_list = reader.read_file()

        # Preprocess the text data
        processor = TextPreprocessor(remove_stopwords=True)
        preprocessed_strings = [processor.preprocess_text(items.get('string', '')) for items in data_list]
        labels = [items.get('label', '') for items in data_list]

        # Fill NA with label confidence with mean value
        label_confidence = [items.get('label_confidence', None) for items in data_list]
        na_counts = label_confidence.count(None)
        mean_confidence = sum([conf for conf in label_confidence if conf is not None]) / (
                len(label_confidence) - na_counts)
        label_confidence = [mean_confidence if conf is None else conf for conf in label_confidence]

        return preprocessed_strings, labels, label_confidence


if __name__ == "__main__":
    eda = EDA()
    preprocessed_strings, labels, label_confidence = eda.get_train(save_train=True)
    tokenized_texts = [text for text in preprocessed_strings]
    longest_sentence = max(tokenized_texts, key=len)

    # Print the first 5 samples
    for i in range(5):
        print("Sample", i + 1)
        print("Preprocessed string:", preprocessed_strings[i])
        print("Label:", labels[i])
        print("Label confidence:", label_confidence[i])
        print()

    # Generate a distribution for label confidence
    plt.hist(label_confidence, bins=20, edgecolor='black')
    plt.title('Distribution of Label Confidence')
    plt.xlabel('Confidence Level')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
