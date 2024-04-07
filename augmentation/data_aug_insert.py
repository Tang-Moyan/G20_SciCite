from transformers import pipeline
import random
import pandas as pd
from transformers import BertTokenizer

unmasker = pipeline('fill-mask', model='bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#this inserts a random word into a random index in the original sentence

def random_insert(string):

    extra_string = ''

    #cannot work if the token length is more than 512, so extract out the extra string first then add it back to the end
    if len(string) > 512:
        extra_string = string[512:]
        string = string[:512]


    input_text = string

    orig_text_list = input_text.split()
    len_input = len(orig_text_list)

    rand_idx = random.randint(1,len_input-2)

    new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx:]
    new_mask_sent = ' '.join(new_text_list)

    augmented_text_list = unmasker(new_mask_sent)
    augmented_text = augmented_text_list[0]['sequence']
    return augmented_text + extra_string

#example of this transformation would be:
#original: I want to eat a banana
#new: I want to go eat delicious banana

df = pd.read_csv('./trainOriginal.csv')

#apply random insert once to method, and thrice sequentially to result, so the results column would have setence with 1, 2 and 3 random inserts. 

method_rows = df[df['label'] == 'method'].copy()
method_rows['string'] = method_rows['string'].apply(random_insert)

result_rows1 = df[df['label'] == 'result'].copy()
result_rows1['string'] = result_rows1['string'].apply(random_insert)

result_rows2 = result_rows1[result_rows1['label'] == 'result'].copy()
result_rows2['string'] = result_rows2['string'].apply(random_insert)

result_rows3 = result_rows2[result_rows2['label'] == 'result'].copy()
result_rows3['string'] = result_rows3['string'].apply(random_insert)

result_df = pd.concat([df, method_rows, result_rows1, result_rows2, result_rows3]).sort_index(kind='merge').reset_index(drop=True)

label_counts = result_df['label'].value_counts()
print(label_counts)

result_df.to_csv('./data_aug/rand_insert_train.csv', index=False, columns=['string', 'label', 'label_confidence'])

