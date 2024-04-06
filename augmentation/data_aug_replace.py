from transformers import pipeline
import random
import pandas as pd
from transformers import BertTokenizer

unmasker = pipeline('fill-mask', model='bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#this replace a random word in the original sentence
def random_replace(string):

    extra_string = ''

    if len(string) > 512:
        extra_string = string[512:]
        string = string[:512]

    input_text = string

    orig_text_list = input_text.split()
    len_input = len(orig_text_list)

    rand_idx = random.randint(1,len_input-1)

    orig_word = orig_text_list[rand_idx]
    new_text_list = orig_text_list.copy()
    new_text_list[rand_idx] = '[MASK]'
    new_mask_sent = ' '.join(new_text_list)

    augmented_text_list = unmasker(new_mask_sent)
    for res in augmented_text_list:
        if res['token_str'] != orig_word:
            augmented_text = res['sequence']
            break
    return augmented_text + extra_string

#example of this transformation would be:
#original: I want to eat a banana
#new: I want to be a banana

df = pd.read_csv('./trainOriginal.csv')

#apply random replace once to method, and thrice sequentially to result, so the results column would have new sentences with 1, 2 and 3 random replaces. 
method_rows = df[df['label'] == 'method'].copy()

method_rows['string'] = method_rows['string'].apply(random_replace)

result_rows1 = df[df['label'] == 'result'].copy()
result_rows1['string'] = result_rows1['string'].apply(random_replace)

result_rows2 = result_rows1[result_rows1['label'] == 'result'].copy()
result_rows2['string'] = result_rows2['string'].apply(random_replace)

result_rows3 = result_rows2[result_rows2['label'] == 'result'].copy()
result_rows3['string'] = result_rows3['string'].apply(random_replace)


result_df = pd.concat([df, method_rows, result_rows1, result_rows2, result_rows3]).sort_index(kind='merge').reset_index(drop=True)

label_counts = result_df['label'].value_counts()
print(label_counts)

result_df.to_csv('./data_aug/rand_replace_train.csv', index=False, columns=['string', 'label', 'label_confidence'])
