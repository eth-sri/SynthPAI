from datasets import load_dataset
from datasets import DatasetDict
import json
import argparse


data_files = {'json': 'synthpai.jsonl'}
dataset = load_dataset("RobinSta/SynthPAI", data_files=data_files)

def load_dataset(file_type):

    if file_type == "HF":

        print(dataset)

        with open('data/thread/synth_clean.jsonl', 'w') as f:
            for record in dataset['json']:
                keys_to_drop = ['children', 'thread_id', 'parent_id'] # drop unused keys
                for key in keys_to_drop:
                    record.pop(key, None)
                if record.get('guesses') is not None:
                    for pred in record['guesses']:
                        if pred.get('feature') == 'age':
                            pred['guesses'] = [int(i) for i in pred['guesses']] # convert age to int format
                json.dump(record, f)
                f.write('\n')
    
    if file_type == "local":

        with open('data/synthpai.jsonl', 'r') as local_file, open('data/thread/synth_clean.jsonl', 'w') as outfile:

            for line in local_file:
                data = json.loads(line)
                keys_to_drop = ['children', 'thread_id', 'parent_id'] # drop unused keys
                for key in keys_to_drop:
                    data.pop(key, None)
                if data.get('guesses') is not None:
                    for pred in data['guesses']:
                        if pred.get('feature') == 'age':
                            pred['guesses'] = [int(i) for i in pred['guesses']] # convert age to int format
                outfile.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file_type",
        type=str,
        help="File type user wants to load - HF for HuggingFace dataset or local for local dataset",
    )
    args = argparser.parse_args()
    load_dataset(args.file_type)