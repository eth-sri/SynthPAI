import json

def process_data(data, gt_data):

    authors = {}
    hardness_mapping = {"complicated": 4, "indirect": 2, "direct": 1}
    features = ["age", "sex", "relationship_status", "education", "occupation", "city_country", "birth_city_country", "income_level"]  # replace with your actual feature list

    for entry in data:

        author = entry["author"]
        if author not in authors:
            authors[author] = {
                "username": entry["username"],
                "author": author,
                "comments": [],
                "predictions": {
                    "human": {feature: {"guess": [""], "hardness": 0, "certainty": 0} for feature in features},
                    "gpt-4": {feature: {"guess": [""], "hardness": 0, "certainty": 0} for feature in features},
                    "ground_truth": {feature: {"guess": [str(gt_data.get(author, {}).get(feature, ""))], "hardness": 0, "certainty": 0} for feature in features}
                },
                "reviews": {
                    "human": {feature: {"estimate": "", "hardness": 0, "certainty": 0} for feature in features},
                    "gpt-4": {feature: {"estimate": "", "hardness": 0, "certainty": 0} for feature in features}
                },
                "evaluations": {}
            }

        authors[author]["comments"].append({"text": entry["text"], "username": entry["username"]})

        for key, value in entry["reviews"]["human"].items():
            if key in features:
                if value["hardness"] > 0 and (value["hardness"] < authors[author]["predictions"]["human"][key]["hardness"] or authors[author]["predictions"]["human"][key]["hardness"] == 0):
                    authors[author]["predictions"]["human"][key]["hardness"] = value["hardness"]
                    authors[author]["predictions"]["human"][key]["certainty"] = value["certainty"]
                    authors[author]["predictions"]["human"][key]["guess"] = [str(value["estimate"])]
                    authors[author]["reviews"]["human"][key]["hardness"] = value["hardness"]
                    authors[author]["reviews"]["human"][key]["certainty"] = value["certainty"]
                    authors[author]["reviews"]["human"][key]["estimate"] = str(value["estimate"])

        authors[author]["reviews"]["human"]["timestamp"] = 0
        authors[author]["reviews"]["human"]["time"] = 0

        if entry["guesses"] is not None:
            for guess in entry["guesses"]:
                if guess["feature"] in features:
                    hardness = hardness_mapping.get(guess["hardness"], 0)
                    if hardness > 0 and (hardness < authors[author]["predictions"]["gpt-4"][guess["feature"]]["hardness"] or authors[author]["predictions"]["gpt-4"][guess["feature"]]["hardness"] == 0):
                        authors[author]["predictions"]["gpt-4"][guess["feature"]]["hardness"] = hardness
                        authors[author]["predictions"]["gpt-4"][guess["feature"]]["certainty"] = guess["certainty"]
                        authors[author]["predictions"]["gpt-4"][guess["feature"]]["guess"] = [str(guess["guesses"][0])]
                        authors[author]["reviews"]["gpt-4"][guess["feature"]]["hardness"] = hardness
                        authors[author]["reviews"]["gpt-4"][guess["feature"]]["certainty"] = guess["certainty"]
                        authors[author]["reviews"]["gpt-4"][guess["feature"]]["estimate"] = str(guess["guesses"][0])

        authors[author]["reviews"]["gpt-4"]["timestamp"] = 0
        authors[author]["reviews"]["gpt-4"]["time"] = 0

    return authors



with open('data/thread/synth_clean.jsonl', 'r') as infile, open('data/profiles/user_bot_gen_online_profiles_300.json', 'r') as gtfile, open('data/thread/comments_label_eval.jsonl', 'w') as outfile:

    data = [json.loads(line) for line in infile]
    gt_data = json.load(gtfile)
    authors = process_data(data, gt_data)

    for author in authors.values():
        outfile.write(json.dumps(author) + '\n')

