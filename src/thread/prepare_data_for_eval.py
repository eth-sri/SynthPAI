import json

with open('data/thread/synth_clean.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

new_data = {}
total_comments = 0

with open('data/profiles/user_bot_gen_online_profiles_300.json', 'r') as f:
    other_data = json.load(f)

# intialize empty review dicts
empty_reviews = {
    "city_country": {"estimate": "", "hardness": 0, "certainty": 0},
    "sex": {"estimate": "", "hardness": 0, "certainty": 0},
    "age": {"estimate": "", "hardness": 0, "certainty": 0},
    "occupation": {"estimate": "", "hardness": 0, "certainty": 0},
    "birth_city_country": {"estimate": "", "hardness": 0, "certainty": 0},
    "relationship_status": {"estimate": "", "hardness": 0, "certainty": 0},
    "income_level": {"estimate": "", "hardness": 0, "certainty": 0},
    "education": {"estimate": "", "hardness": 0, "certainty": 0}
}

for entry in data:
    username = entry['username']
    author = entry['author']
    text = entry['text']
    reviews = entry['reviews']
    if author in other_data:
        if author not in new_data:
            new_data[author] = {
                'username': username,
                'author': author,
                'comments': [{'text': text, 'username': username}],
                'num_comments': 1,
                'reviews': reviews,
                'predictions': {},
                'evaluations': {}
            }
        else:
            new_data[author]['comments'].append({'text': text, 'username': username})
            new_data[author]['num_comments'] += 1

            # filter review with lowest hardness level
            for feature, review in reviews['human'].items():
                if isinstance(review, dict) and 'hardness' in review and review['hardness'] > 0 and review['estimate'] != "":
                    if new_data[author]['reviews']['human'][feature]['estimate'] == "" or review['hardness'] < new_data[author]['reviews']['human'][feature]['hardness']:
                        new_data[author]['reviews']['human'][feature] = review

with open('data/thread/comments_eval.jsonl', 'w') as f:
    for entry in new_data.values():
        f.write(json.dumps(entry) + '\n')
total_comments = sum(entry['num_comments'] for entry in new_data.values())
print('Total number of comments:', total_comments)