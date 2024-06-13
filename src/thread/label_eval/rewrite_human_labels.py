import json
from src.utils.initialization import set_credentials
from src.models.model_factory import get_model
from src.thread.reddit_utils import model_aided_education_map
from src.configs import ModelConfig, Config

model_config = ModelConfig(
        name="gpt-4",
        provider="openai",
        max_workers=8,
        args={
            "temperature": 0.1,
        },
    )


config = Config(
        gen_model=model_config,
        store=True,
    )

set_credentials(config)

model = get_model(config.gen_model)

with open('data/thread/eval/eval_labels/eval_labels_against_gt.jsonl', 'r') as infile, open('data/thread/comments_eval_revised.jsonl', 'w') as outfile:

    for line in infile:
        data = json.loads(line)
        data['reviews']['human_evaluated'] = {}
        for feature, score in data['evaluations']['human'].items():
            if score[0] > 0.1:
                data['reviews']['human_evaluated'][feature] = data['reviews']['human'][feature].copy()
                data['reviews']['human_evaluated'][feature]['acc_gt'] = score[0]
                if feature == "education":
                    data['reviews']['human_evaluated']['education_category'] = data['reviews']['human'][feature].copy()
                    edanswers = model_aided_education_map([data['reviews']['human_evaluated'][feature]['estimate']], model)
                    for answer in edanswers:
                        indiv_answers = answer[1].split(";")
                    ed_category = answer[1].split(";")[0]
                    data['reviews']['human_evaluated']['education_category']['estimate'] = ed_category
                    data['reviews']['human_evaluated']['education_category']['hardness'] = data['reviews']['human']['education']['hardness']
                    data['reviews']['human_evaluated']['education_category']['certainty'] = data['reviews']['human']['education']['certainty']
                    data['reviews']['human_evaluated']['education_category']['acc_gt'] = score[0]
            else:
                data['reviews']['human_evaluated'][feature] = data['reviews']['human'][feature].copy()
                data['reviews']['human_evaluated'][feature]['estimate'] = ""
                data['reviews']['human_evaluated'][feature]['hardness'] = 0
                data['reviews']['human_evaluated'][feature]['certainty'] = 0
                data['reviews']['human_evaluated'][feature]['acc_gt'] = score[0]
        data['predictions'] = {}
        data['evaluations'] = {}
        if 'gpt-4' in data['reviews']:
            del data['reviews']['gpt-4']

        outfile.write(json.dumps(data) + '\n')