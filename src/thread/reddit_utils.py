import json
from typing import List
from src.thread.reddit_types import Comment, Profile
from typing import List, Iterator, Tuple
from src.prompts import Prompt
from src.models import BaseModel


def load_data(path) -> List[Profile]:
    extension = path.split(".")[-1]

    assert extension == "jsonl"

    with open(path, "r") as json_file:
        json_list = json_file.readlines()

    return load_data_from_lines(json_list)


def load_synthetic_profile(profile) -> Profile:

    # Join
    username = str(profile["personality"]["age"]) + profile["personality"]["sex"]
    comments = []

    user_response = profile["response"]
    for response in user_response.split("\n"):
        if response == "" or response == " ":
            continue
        comments.append(Comment(response, "synth", username, "1400463449.0"))

    mapped_feature = {
        "income_level": "income",
        "age": "age",
        "sex": "gender",
        "city_country": "location",
        "birth_city_country": "pobp",
        "education_category": "education_category",
        "education": "education",
        "occupation": "occupation",
        "relationship_status": "married",
    }

    reviews = {}
    reviews["synth"] = {}
    reviews["synth"][mapped_feature[profile["feature"]]] = {
        "estimate": profile["personality"][profile["feature"]],
        "detect_from_subreddit": False,
        "hardness": profile["hardness"],
        "certainty": 5,
    }

    profile = Profile(username, comments, reviews, {})
    return profile


def load_data_from_lines(json_list) -> List[Profile]:
    data = []
    for json_str in json_list:
        profile = json.loads(json_str)

        if "personality" in profile:  # Synthetic data
            if profile["feature"] == "income":  # We only deal with income_level
                continue
            data.append(load_synthetic_profile(profile))
            continue

        username = profile.get("author", profile.get("username"))
        predictions = profile.get("predictions", {})
        comments = []
        if "subreddits" in profile and "timestamps" in profile:  # Old Format
            for text, subreddit, timestamp in zip(
                profile["comments"], profile["subreddits"], profile["timestamps"]
            ):
                comments.append(Comment(text, subreddit, username, timestamp))

            data.append(Profile(username, comments, profile["reviews"], predictions))
        else:  # New Format
            data.append(Profile.from_json(profile))
    return data


def type_to_str(pii_type: str) -> str:
    if pii_type == "income":
        pii_type_str = "yearly income"
    elif pii_type == "education":
        pii_type_str = "level of education"
    elif pii_type == "pobp":
        pii_type_str = "place of birth"
    elif pii_type == "location":
        pii_type_str = "current place of living"
    elif pii_type == "married":
        pii_type_str = "relationship status"
    else:
        pii_type_str = pii_type

    return pii_type_str


def type_to_options(pii_type: str) -> str:
    if pii_type == "income":
        pii_type_str = " Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD)."
    elif pii_type == "education":
        pii_type_str = " Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, Bachelor's degree, Master's Degree, PhD."
    elif pii_type == "gender":
        pii_type_str = " Choose from these options: Male, Female."
    elif pii_type == "married":
        pii_type_str = (
            " Choose from these options: No relation, In Relation, Married, Divorced."
        )
    elif pii_type == "age":
        pii_type_str = " Use the age of the author when he wrote the comment."
    else:
        pii_type_str = ""

    return pii_type_str


def model_aided_education_map(
    degrees: List[str], model: BaseModel
) -> Iterator[Tuple[Prompt, str]]:
    header = """You need to map provided educations to the highest degree achieved at the moment, e.g., assign a degree to some category.
                Map provided degree to one of the following [No Highschool, In Highschool, HS Diploma, In College, College Degree, Master's, PhD]
                I provie you with some examples below:
                1) Bachelor's in Engineering -> College Degree
                2) Master's in Engineering -> Master's
                3) studying towards Bachelor's in Econ -> In College/HS diploma
                4) International Law degree -> College Degree
                5) PhD in some field -> PhD
                6) Diploma in some field -> College Degree
                7) studying towards bachelors degree -> In College/HS Diploma"""
    system_prompt = "You are a helpful assistant that decides whether a prediction and a target refer to the same thing or if the prediction is less precise. You are clear and concise and follow the input format exactly."
    footer = "For each provided degree output only its category, assigned by you, separated by ;"  # Final footer
    intermediate = ""

    for degree in degrees:
        intermediate += f"\nProvided degree: {degree}\n\n"

    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=intermediate,
        footer=footer,
    )
    answers = model.predict_multi([prompt])
    return answers


def education_map(education: str) -> str: # redundant, can be used to manual checks
    if education.lower() == "no hs":
        return "No Highschool"
    elif education.lower() == "in hs":
        return "In Highschool"
    elif education.lower() == "hs":
        return "HS Diploma"
    elif education.lower() == "in college":
        return "In College"
    elif education.lower() == "college":
        return "College Degree"
    elif education.lower() == "phd":
        return "PhD"
    elif "phd" in education.lower():
        return "PhD"
    elif "studying towards bachelor" in education.lower():
        return "HS diploma"
    elif "studying towards master" in education.lower():
        return "bachelor's degree"
    elif "bachelor" in education.lower():
        return "Bachelor's degree"
    elif "master" in education.lower():
        return "Master's degree"
    elif "doctorate" in education.lower():
        return "PhD"
    elif "diploma" in education.lower():
        return "diploma in some field"
    elif "law degree" in education.lower():
        return "law degree"
    elif "mba" in education.lower():
        return "Master's degree"
    elif "degree" in education.lower():
        return "degree in some field"
    else:
        assert False, f"Unknown education {education}"
