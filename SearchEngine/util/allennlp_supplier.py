import subprocess
import json
import os

def retrieve_allennlp_predictions(id_string_list):
    '''
    This function takes in data of the form (id, string) and tries to predict if it is a background, method or result using the allennlp model.

    This will spawn a subprocess to run the allennlp predictor and block until the subprocess is finished. It will then return the predictions in the form of (id, prediction) where prediction is either "background", "method" or "result".
    '''
    # Save the current working directory
    cwd = os.getcwd()

    # For each string, format it in the jsonl format:
    formatted_strings = [{"source": "explicit", "citeEnd": 175, "sectionName": "Introduction", "citeStart": 168, "string": string, "label": "background", "label_confidence": 1.0, "citingPaperId": str(id), "citedPaperId": str(id), "isKeyCitation": "true", "id": str(id), "unique_id": str(id), "excerpt_index": 11} for id, string in id_string_list]

    # Dump all the formatted strings into a jsonl file
    with open(f"{cwd}/formatted_strings.jsonl", "w") as f:
        for formatted_string in formatted_strings:
            f.write(json.dumps(formatted_string) + "\n")

    # Using a subprocess, open the allennlp predictor and pass the jsonl file to it
    # Block until the subprocess is finished
    subprocess.run(["python", "-m", "allennlp.run", "predict", "C:\\Users\\ruihan\\Downloads\\scicite\\scicite-pretrained.tar.gz", f"{cwd}/formatted_strings.jsonl", "--predictor", "predictor_scicite", "--include-package", "scicite", "--overrides", "{'model':{'data_format':''}}", "--output-file", f"{cwd}/out.jsonl"], cwd="..")

    # Read the output file and return the id and predictions by taking only the "citingpaperid" and "prediction" field of each jsonl object
    with open(f"{cwd}/out.jsonl", "r") as f:
        predictions = []
        for line in f:
            json_obj = json.loads(line)
            predictions.append((json_obj["citingPaperId"], json_obj["prediction"]))

    return predictions