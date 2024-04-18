# Merge the dev, test, and train data into a single file for AllenNLP

import argparse
import json
import os

def merge_allennlp_data(data_dir, output_file):
    print("Data dir: ", data_dir)
    print("Output file: ", output_file)
    data = []
    for split in ['train', 'dev', 'test']:
        with open(os.path.join(data_dir, f"{split}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

    # Remove duplicates
    data = [dict(t) for t in {tuple(d.items()) for d in data}]
    
    with open(output_file, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the train, dev, and test data files")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to write the merged data to")
    args = parser.parse_args()

    merge_allennlp_data(args.data_dir, args.output_file)