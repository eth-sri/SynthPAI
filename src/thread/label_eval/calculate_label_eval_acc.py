import json
from collections import defaultdict

def calculate_acc(eval_type):

    total_scores_human = 0
    total_scores_gpt4 = 0
    total_scores_gt = 0
    count_human = 0
    total_scores_human_positive = 0
    count_human_lp = defaultdict(int) # count less precise
    count_human_tag = defaultdict(int)
    count_human_positive = 0
    count_gpt4 = 0
    count_gt = 0
    # count only precise labels next
    total_scores_human_01 = 0
    total_scores_gpt4_01 = 0
    total_scores_gt_01 = 0
    count_human_01 = 0
    count_gpt4_01 = 0
    count_gt_01 = 0


    if eval_type == "gt":
        print("===Eval against ground truth====")
        file = open('data/thread/eval/eval_labels/eval_labels_against_gt.jsonl', 'r')
    if eval_type == "model":
        print("===Eval against model labels====")
        file = open('data/thread/eval/eval_labels/eval_labels_against_model.jsonl', 'r')
    if eval_type == "human":
        print("===Eval against human labels====")
        file = open('data/thread/eval/eval_labels/eval_labels_against_human.jsonl', 'r')


    for line in file:

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            continue


        if 'human' in data['evaluations']:
            evaluations_human = data['evaluations']['human']
            if eval_type == "model":
                evaluations_human = data['evaluations']['human']['gpt-4']
            for key in evaluations_human:
                total_scores_human += evaluations_human[key][0]
                count_human += 1
                count_human_positive += 1
                count_human_01 += 1
                
                if evaluations_human[key][0] > 0.1:
                    if evaluations_human[key][0] == 1:
                        total_scores_human_positive += evaluations_human[key][0]
                        count_human_tag[key] += 1
                
                if evaluations_human[key][0] in [0, 1]:
                        total_scores_human_01 += evaluations_human[key][0]
                
                if evaluations_human[key][0] == 0.5:
                    count_human_lp[key] += 1

        if 'gpt-4' in data['evaluations']:
            evaluations_gpt4 = data['evaluations']['gpt-4']
            if eval_type == "human":
                evaluations_gpt4 = data['evaluations']['gpt-4']['human']
            for key in evaluations_gpt4:
                total_scores_gpt4 += evaluations_gpt4[key][0]
                count_gpt4 += 1
                count_gpt4_01 += 1

                if evaluations_gpt4[key][0] in [0, 1]:
                        total_scores_gpt4_01 += evaluations_gpt4[key][0]

        if 'ground_truth' in data['evaluations']:
            evaluations_gt = data['evaluations']['ground_truth']
            if eval_type == "model":
                evaluations_gt = data['evaluations']['ground_truth']['gpt-4']
            if eval_type == "human":
                evaluations_gt = data['evaluations']['ground_truth']['human']
            for key in evaluations_gt:
                total_scores_gt += evaluations_gt[key][0]
                if evaluations_gt[key][0] in [0, 1]:
                        total_scores_gt_01 += evaluations_gt[key][0]
                count_gt += 1
                count_gt_01 += 1
                

    if count_human > 0:
        average_score_human = round(total_scores_human / count_human, 2)
        average_score_human_positive = round(total_scores_human_positive / count_human_positive, 2) if count_human_positive > 0 else 0
        average_score_human_01 = round(total_scores_human_01 / count_human_01, 2) if count_human_01 > 0 else 0
        print('Average reviewer guessing accuracy for "human":', average_score_human*100, "%")
        print('revised label acc: ', average_score_human_positive)
        print('0-1 accuracy for "human":', average_score_human_01*100, '%')
    if count_gpt4 > 0:
        average_score_gpt4 = round(total_scores_gpt4 / count_gpt4, 2)
        average_score_gpt4_01 = round(total_scores_gpt4_01 / count_gpt4_01, 2) if count_gpt4_01 > 0 else 0
        print('Average reviewer guessing accuracy for "gpt-4":', average_score_gpt4*100, "%")
        print('0-1 accuracy for "gpt-4":', average_score_gpt4_01*100, '%')
    if count_gt > 0:
        average_score_gt = round(total_scores_gt / count_gt, 2)
        average_score_gt_01 = round(total_scores_gt_01 / count_gt_01, 2) if count_gt_01 > 0 else 0
        print('Average reviewer guessing accuracy for "ground_truth":', average_score_gt*100, "%")
        print('0-1 accuracy for "ground_truth":', average_score_gt_01*100, '%')

def guess_count():

    both_non_empty = 0
    guesses_non_empty = 0
    human_non_empty = 0
    both_empty = 0

    with open('data/thread/synth_clean.jsonl', 'r') as f:

        for line in f:

            data = json.loads(line)

            guesses = data.get('guesses', [])
            human = data.get('reviews', {}).get('human', {})
            human = {k: v for k, v in human.items() if k not in ['time', 'timestamp']}
            guesses_dict = {guess.get('feature'): guess.get('guesses', []) for guess in guesses} if guesses else {}

            for feature, value in human.items():
                guess_values = guesses_dict.get(feature, [])
                human_estimate = value.get('estimate')

                if guess_values and human_estimate:
                    both_non_empty += 1
                elif guess_values and not human_estimate:
                    guesses_non_empty += 1
                elif not guess_values and human_estimate:
                    human_non_empty += 1
                elif not guess_values and not human_estimate:
                    both_empty += 1

    print(f'Both non-empty: {both_non_empty}')
    print(f'Model guesses non-empty, human empty: {guesses_non_empty}')
    print(f'Model guesses empty, human non-empty: {human_non_empty}')
    print(f'Both empty: {both_empty}')

    total = both_non_empty + guesses_non_empty + human_non_empty + both_empty
    fpr = round((guesses_non_empty / (guesses_non_empty + both_empty) if total else 0) * 100, 2)
    fnr = round((human_non_empty / (human_non_empty + both_non_empty) if total else 0)*100, 2)
    print(f'False positive rate: {fpr}%')
    print(f'False negative rate: {fnr}%')


if __name__ == "__main__":
    calculate_acc("gt")
    calculate_acc("model")
    calculate_acc("human")
    print("===Label count===")
    guess_count()