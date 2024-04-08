import csv
import json
import glob
import os

def convert(src, csv_file):

    jsonl_file = csv_file.replace('.csv', '.jsonl')
    csv_strings = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            csv_strings.append(row[0])



    matched_dicts = []
    with open(src, 'r', encoding='utf-8') as jsonlfile:
        for line in jsonlfile:
            data = json.loads(line)
            if data['string'] in csv_strings:
                matched_dicts.append(data)

 
    with open(jsonl_file, 'w') as outputfile:
        for matched_dict in matched_dicts:
            outputfile.write(json.dumps(matched_dict) + '\n')

src = '../data/train.jsonl'
csv_root = 'data_files/'


csv_files = glob.glob(os.path.join(csv_root, '*.csv'))
for csv_file in csv_files:
    convert(src, csv_file)
