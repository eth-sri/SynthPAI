import os
import json
import pandas as pd
import string
import random

features = ["city_country", "age", "income_level", "income", "education", "occupation", "sex", "relationship_status", "birth_city_country"]
texts_by_person = {}
total_comments = []

def generate_id(length):

    rand_str = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(rand_str) for i in range(length))
    return result_str

def process_entry(entry):

    person = entry['author']
    text = entry['text']
    guesses = entry['guesses']
    
    if person not in texts_by_person:
        texts_by_person[person] = []
    
    comment = []
    comment.append(text)
    comment.append(guesses)
    total_comments.append(1)
    
    texts_by_person[person].append(comment)

    for child in entry['children']:
        process_entry(child)

for topic in features:
    
    for no in range(15):
        
        file = os.path.join(f'data/thread/generated_threads/json_threads/{topic}/', f'thread_{topic}_{no+1}.json')    
        if not os.path.exists(file):
            continue       
        with open(file, 'r') as f:
            data = json.load(f)        
        process_entry(data)

for person, texts in texts_by_person.items():
    
    output_file = os.path.join('data/thread/gens_per_profile', f'{person}_gen_text.json')   
    output_data = {
        'person': person,
        'text': texts
    }    
    with open(output_file, 'w') as f:
        json.dump(output_data, f)

no_com = [] # check profiles/comments without comments

for t in range(300):

    person = f'pers{t+1}' 
    path = f'data/thread/gens_per_profile/{person}_gen_text.json'
    if os.path.isfile(path):
        e = 0
    else:
        no_com.append(person)

print('Profiles without comments: ', len(no_com))

def extract_data_per_guess(data, result = [], result_labeled=[]):
    

    if len(data['guesses']) == 0:
        row = {}
        row['author'] = data['author']
        row['profile'] = data['profile']
        row['text'] = data['text']
        row['guess_feature'] = None
        row['guesses'] = None
        row['hardness'] = None
        row['certainty'] = None
        row['model_eval_top1'] = None
        row['model_eval_top3'] = None
        row['model_eval'] = None
        result.append(row)
    else:
        for guess in data['guesses']:
            row = {}
            row['author'] = data['author']
            row['profile'] = data['profile']
            row['text'] = data['text']
            row['guess_feature'] = guess['feature']
            row['guesses'] = guess['guesses']
            row['hardness'] = guess['hardness']
            row['certainty'] = guess['certainty']
            row['model_eval_top1'] = guess['model_eval'][0]
            row['model_eval_top3'] = max(guess['model_eval'])
            row['model_eval'] = guess['model_eval']
            result.append(row)
            result_labeled.append(row)
    for child in data['children']:
        extract_data_per_guess(child, result, result_labeled)

    return result, result_labeled

def extract_data(data, result = []):


    if len(data['guesses']) == 0:
        row = {}
        row['author'] = data['author']
        row['username'] = data['profile']['username']
        row['profile'] = data['profile']
        row['text'] = data['text']
        row['guesses'] = None
        row['id'] = generate_id(10)
        result.append(row)
    else:
        row = {}
        row['author'] = data['author']
        row['username'] = data['profile']['username']
        row['profile'] = data['profile']
        row['text'] = data['text']
        row['guesses'] = data['guesses']
        row['id'] = generate_id(10)
        result.append(row)
    with open('data/thread/comments_unique_id.jsonl', 'a+') as f:
        f.write(json.dumps(row) + '\n')
    for child in data['children']:
        extract_data(child, result)

    return result

df = pd.DataFrame()
dfs = []
df_labeled = pd.DataFrame()
dfs_labeled = []

for topic in features:
    for i in range(50):
        filename = f"data/thread/generated_threads/json_threads/{topic}/thread_{topic}_{i}.json"

        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                data_json = json.load(f)           
            data, labeled_data = extract_data_per_guess(data_json)
            data_plain = extract_data(data_json) # to save unique comments with all guesses
            dfs.append(pd.DataFrame(data))
            dfs_labeled.append(pd.DataFrame(labeled_data))

df = pd.concat(dfs, ignore_index=True)
df.to_csv('data/thread/comments.csv', index=False)
df = pd.read_csv('data/thread/comments.csv') 
df = df.drop_duplicates()
# fix profile as dict
df['profile'] = df['profile'].map(lambda x: dict(eval(x)) if pd.notnull(x) else x)
print('Total comments: ', df["text"].nunique())
df.to_csv('data/thread/comments.csv', index=False)

# convert to jsonl for grading pipeline
with open("data/thread/comments.jsonl", "w", encoding='utf8') as f:
    f.write(df.to_json(orient='records', lines=True, force_ascii=False))

df_labeled = pd.concat(dfs_labeled, ignore_index=True)
df_labeled.to_csv('data/thread/labeled_comments.csv', index=False)
df_labeled = pd.read_csv('data/thread/labeled_comments.csv') 
df_labeled = df_labeled.drop_duplicates()
# fix profile desc as dict
df_labeled['profile'] = df_labeled['profile'].map(lambda x: dict(eval(x)) if pd.notnull(x) else x)
print('Labeled comments: ', df_labeled["text"].nunique())
df_labeled.to_csv('data/thread/labeled_comments.csv', index=False)

with open("data/thread/labeled_comments.jsonl", "w", encoding='utf8') as f:
    f.write(df_labeled.to_json(orient='records', lines=True, force_ascii=False))