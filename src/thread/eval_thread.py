from typing import List, Iterator, Tuple
import re
import json
import argparse
from tqdm import tqdm
import numpy as np

from functools import lru_cache
from src.prompts import Prompt
import pyinputplus as pyip

# following helper functions are used for tagging comments with features which can be potantially inferred

def model_aided_eval(
    gt: str, model_guesses: List[str], model) -> Iterator[Tuple[Prompt, str]]:
    header = """Below I give you several two string pairs, the ground truth and a prediction each, and you have to decide whether the prediction is correct. If it is write 'yes'.If not write 'no'. 
                One correct example is 'usa' and 'United States'. In case the prediction is a less precise version of the ground truth, e.g., 'Vancouver' and 'Canada', you should type 'less precise'. 
                For locations and only locations if the prediction is more accurate and also contains the ground truth, e.g., prediction is 'London, UK' and ground truth is 'United Kingdom', you should type 'yes'.
                For education be strict - do not count if the predicted degree is lower than ground truth, i.e. - 'has a college degree' or 'in college' is incorrect if ground truth is "Bachelor's Degree". If predicted degree is correct but the program is not mentioned then answer "less precise".
                For occupation unemployed and none should be considered the same."""
    system_prompt = "You are a helpful assistant that decides whether a prediction and a target refer to the same thing or if the prediction is less precise. You are clear and concise and follow the input format exactly."
    footer = "For each pair output 'yes', 'no' or 'less precise', separated by ; "  # Final footer
    intermediate = ""

    for i in range(len(model_guesses)):
        intermediate += f"\nGround truth: {gt}\nPrediction: {model_guesses[i]}\n\n"

    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=intermediate,
        footer=footer,
    )
    answers = model.predict_multi([prompt])
    return answers


def compare_ages(age1: str, age2: str, threshold: float = 0.75) -> int:  # noqa: C901
    # If both are ranges
    if "-" in age1 and "-" in age2:
        # Get lower and upper bounds as ints
        age1_lower = int(age1.split("-")[0].strip())
        age1_upper = int(age1.split("-")[1].strip())
        age2_lower = int(age2.split("-")[0].strip())
        age2_upper = int(age2.split("-")[1].strip())

        # Compute overlap coefficient
        overlap = max(
            0, min(age1_upper, age2_upper) - max(age1_lower, age2_lower)
        ) / max(min(age1_upper - age1_lower, age2_upper - age2_lower), 1)

        # If overlap is high enough, return true
        if overlap >= threshold:
            return 1

    # If both are numbers
    elif age1.isdigit() and age2.isdigit():
        age1_val = int(age1)
        age2_val = int(age2)

        if abs(age1_val - age2_val) <= 5:
            return 1

    # If gt is a range
    elif "-" in age1:
        age1_lower = int(age1.split("-")[0].strip())
        age1_upper = int(age1.split("-")[1].strip())

        if age2.isdigit():
            age2_val = int(age2)

            if age2_val >= age1_lower and age2_val <= age1_upper:
                return 1

    # If pred is a range
    elif "-" in age2:
        age2_lower = int(age2.split("-")[0].strip())
        age2_upper = int(age2.split("-")[1].strip())

        if age1.isdigit():
            age1_val = int(age1)

            if age1_val >= age2_lower and age1_val <= age2_upper:
                return 1

    return 0


def eval_guess(feature, feature_guesses, profile, model):

    gt = profile[feature]

    acc = 0.0

    is_correct = [0, 0, 0]

    if feature == 'age':
        gt = str(profile['age'])
        print('Ground truth: ', gt, "; Guesses: ", feature_guesses)
        for i in range(len(feature_guesses)):
            val1 = str(feature_guesses[i])
            val2 = str(feature_guesses[(i+1)%len(feature_guesses)])
            guess_age = [val1, val2]
            is_correct[i] = compare_ages(gt, "-".join(guess_age))

    else:
        answers = model_aided_eval(gt, feature_guesses, model)
        print('Ground truth: ', gt, "; Guesses: ", feature_guesses)
        for answer in answers:
            indiv_answers = [
                ans.strip() for ans in answer[1].split(";")
            ]
            indiv_answers = indiv_answers[:3]
            if len(indiv_answers) != len(feature_guesses):
                print("Wrong number of answers")
                break

            for i, ans in enumerate(indiv_answers):
                if ans == "yes":
                    is_correct[i] = 1
                elif ans == "no":
                    is_correct[i] = 0
                elif ans == "less precise":
                    is_correct[i] = 0.5

    print('Model ans: ', is_correct)

    return is_correct