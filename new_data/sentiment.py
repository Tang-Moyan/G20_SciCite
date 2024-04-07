from textblob import TextBlob
import pandas as pd

def sentiment_extraction(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def sentiment_words(num):
    if num > 0:
        return 'Positive'
    if num < 0:
        return 'Negative'
    return 'Neutral'

csv_file_path = './trainOriginal.csv'  

df = pd.read_csv(csv_file_path)


df['sentiment'] = df['string'].apply(sentiment_extraction)

df['sentiment_word'] = df['sentiment'].apply(sentiment_words)

label_counts = df['sentiment_word'].value_counts()
print(label_counts)

label_sentiment_counts = df.groupby('sentiment_word')['label'].value_counts()
print(label_sentiment_counts)

label_sentiment_counts = df.groupby('label')['sentiment_word'].value_counts()
print(label_sentiment_counts)

# print(df.columns)

df_positive = df[df['sentiment_word'] == 'Positive']
df_neutral = df[df['sentiment_word'] == 'Neutral']
df_negative = df[df['sentiment_word'] == 'Negative']

df.to_csv('./data_files/overall_sentiment_train.csv', index=False)
df_positive.to_csv('./data_files/postive_train.csv', index=False)
df_neutral.to_csv('./data_files/neutral_train.csv', index=False)
df_negative.to_csv('./data_files/negative_train.csv', index=False)

# Positive    3429
# Neutral     2819
# Negative    1995
