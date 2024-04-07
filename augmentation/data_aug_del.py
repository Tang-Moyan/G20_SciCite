import random
import pandas as pd
def random_deletion(words, p):
    #with a probability of p deleting each word from the sentence
    words = words.split()
    
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)
    
    return sentence


csv_file_path = './trainOriginal.csv' 


df = pd.read_csv(csv_file_path)

method_rows = df[df['label'] == 'method'].copy()
result_rows1 = df[df['label'] == 'result'].copy()
result_rows2 = df[df['label'] == 'result'].copy()
result_rows3 = df[df['label'] == 'result'].copy()
result_rows4 = df[df['label'] == 'result'].copy()
print("reached here 3")

# apply random delete to method once, then apply thrice to results with different probability p (can modify)
method_rows['string'] = method_rows['string'].apply(lambda x: random_deletion(x, 0.2))
result_rows1['string'] = result_rows1['string'].apply(lambda x: random_deletion(x, 0.2))
result_rows2['string'] = result_rows2['string'].apply(lambda x: random_deletion(x, 0.1))
result_rows3['string'] = result_rows3['string'].apply(lambda x: random_deletion(x, 0.3))

result_df = pd.concat([df, method_rows, result_rows1, result_rows2, result_rows3]).sort_index(kind='merge').reset_index(drop=True)

label_counts = result_df['label'].value_counts()
print(label_counts)

result_df.to_csv('./data_aug/rand_del_train.csv', index=False, columns=['string', 'label', 'label_confidence'])
