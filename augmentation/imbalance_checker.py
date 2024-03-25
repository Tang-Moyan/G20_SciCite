
import pandas as pd

train_data = pd.read_json('data/train.jsonl', lines=True)
test_data = pd.read_json('data/test.jsonl', lines=True)
dev_data = pd.read_json('data/dev.jsonl', lines=True)

print('Unique labels:')
print(train_data['label'].unique())

def check(type, data):

    if type == 'train':
        print('\n' + 'TRAIN' + '\n')
    if type == 'test':
        print('\n' + 'TEST' + '\n')
    if type == 'dev':
        print('\n' + 'DEV' + '\n')

    background = data[data['label'] == 'background']
    method = data[data['label'] == 'method']
    result = data[data['label'] == 'result']

    print('background')
    print('count: ' + str(background['label'].count()))
    print('avg_confidence: ' + str(background['label_confidence'].mean()))

    print('\n')
    print('method')
    print('count: ' + str(method['label'].count()))
    print('avg_confidence: ' + str(method['label_confidence'].mean()))
    
    print('\n')
    print('result')
    print('count: ' + str(result['label'].count()))
    print('avg_confidence: ' + str(result['label_confidence'].mean()))

check('train', train_data)
check('test', test_data)
check('dev', dev_data)