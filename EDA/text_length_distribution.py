import matplotlib.pyplot as plt

def text_length_distribution(texts):
    text_lengths = [len(text) for text in texts]
    plt.hist(text_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.grid(True)
    plt.show()
