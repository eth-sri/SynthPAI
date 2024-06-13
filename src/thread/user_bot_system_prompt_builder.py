def build_user_bot_system_prompt(prompt_skeleton: str, profile: dict, writing_style: str, min_comment_len, max_comment_len, ct) -> str:
    """
    Takes a parameterized prompt skeleton for the user bot and a profile dictionary
    and return the filled out prompt skeleton, ready to use for the bot.

    :param prompt_skeleton: A string with parameters <param> matching the keys in the dictionary,
        This skeleton is filled out to generate the system prompt.
    :param profile: A dictionary containing the personal information of the profile that is to be
        impersonated by the user bot.
    :return: Returns the specific persionalized user bot system prompt as a string.
    """
    final_prompt = prompt_skeleton
    for key, value in profile.items():
        if key == 'education':
            if value.startswith('studying'):
                final_prompt = final_prompt.replace(f'<{key}>', f'are {value}')
            else:
                final_prompt = final_prompt.replace(f'<{key}>', f'hold a {value}')
        elif key == 'city_country':
            city, country = value.split(', ')
            final_prompt = final_prompt.replace(f'<city>', str(city))
            final_prompt = final_prompt.replace(f'<country>', str(country))
        else:
            final_prompt = final_prompt.replace(f'<{key}>', str(value))
        final_prompt = final_prompt.replace('<writing_style>', str(writing_style))
        final_prompt = final_prompt.replace('<min_comment_len>', str(min_comment_len))
        final_prompt = final_prompt.replace('<max_comment_len>', str(max_comment_len))
        if ct == True:
            final_prompt = final_prompt.replace('<critic_type>', 'You are always very critical and disagreeing with others there.')
        else:
            final_prompt = final_prompt.replace('<critic_type>', '')
    return final_prompt

def build_author_bot_system_prompt(prompt_skeleton: str, profile: dict) -> str:
    """
    Takes a parameterized prompt skeleton for the user bot and a profile dictionary
    and return the filled out prompt skeleton, ready to use for the bot.

    :param prompt_skeleton: A string with parameters <param> matching the keys in the dictionary,
        This skeleton is filled out to generate the system prompt.
    :param profile: A dictionary containing the personal information of the profile that is to be
        impersonated by the user bot.
    :return: Returns the specific persionalized user bot system prompt as a string.
    """
    final_prompt = prompt_skeleton
    for key, value in profile.items():
        if key == 'education':
            if value.startswith('studying'):
                final_prompt = final_prompt.replace(f'<{key}>', f'are {value}')
            else:
                final_prompt = final_prompt.replace(f'<{key}>', f'hold a {value}')
        elif key == 'city_country':
            city, country = value.split(', ')
            final_prompt = final_prompt.replace(f'<city>', str(city))
            final_prompt = final_prompt.replace(f'<country>', str(country))
        else:
            final_prompt = final_prompt.replace(f'<{key}>', str(value))
    return final_prompt


def build_checker_bot_system_prompt(prompt_skeleton: str, profile: dict, topic: str) -> str:
    """
    Takes a parameterized prompt skeleton for the user bot and a profile dictionary
    and return the filled out prompt skeleton, ready to use for the bot.

    :param prompt_skeleton: A string with parameters <param> matching the keys in the dictionary,
        This skeleton is filled out to generate the system prompt.
    :param profile: A dictionary containing the personal information of the profile that is to be
        impersonated by the user bot.
    :return: Returns the specific persionalized user bot system prompt as a string.
    """
    final_prompt = prompt_skeleton
    for key, value in profile.items():
        if key == 'education':
            if value.startswith('studying'):
                final_prompt = final_prompt.replace(f'<{key}>', f'are {value}')
            else:
                final_prompt = final_prompt.replace(f'<{key}>', f'hold a {value}')
        elif key == 'city_country':
            city, country = value.split(', ')
            final_prompt = final_prompt.replace(f'<city>', str(city))
            final_prompt = final_prompt.replace(f'<country>', str(country))
        else:
            final_prompt = final_prompt.replace(f'<{key}>', str(value))
    final_prompt = final_prompt.replace('<topic>', str(topic))
    return final_prompt


def build_writing_style_prompt(prompt_skeleton: str, profile: dict) -> str:

    final_prompt = prompt_skeleton
    for key, value in profile.items():
        if key == 'education':
            if value.startswith('studying'):
                final_prompt = final_prompt.replace(f'<{key}>', f'are {value}')
            else:
                final_prompt = final_prompt.replace(f'<{key}>', f'hold a {value}')
        elif key == 'city_country':
            city, country = value.split(', ')
            final_prompt = final_prompt.replace(f'<city>', str(city))
            final_prompt = final_prompt.replace(f'<country>', str(country))
        else:
            final_prompt = final_prompt.replace(f'<{key}>', str(value))
    
    return final_prompt


def build_comment_path_prompt(path: list, pers: str) -> str:
    """
    Takes a parameterized prompt skeleton for the user bot and a profile dictionary
    and return the filled out prompt skeleton, ready to use for the bot.

    :param prompt_skeleton: A string with parameters <param> matching the keys in the dictionary,
        This skeleton is filled out to generate the system prompt.
    :param profile: A dictionary containing the personal information of the profile that is to be
        impersonated by the user bot.
    :return: Returns the specific persionalized user bot system prompt as a string.
    """
    
    path.reverse()
    final_prompt = '\n You are ' + pers + '. You are engaging in a Reddit thread and you are replying to its subthread of comments, in particular the last comment. \
    You SHOULD NOT use agreeing/disagreeing phrases directly in your reply, cut straight to the chase. \n \
    You might have possibly already engaged in it before, so you should reply accordingly to be coherent in your backstory and the content you wrote before. \n \
    You should not use emojis and hashtags. Remember to disagree if you were described as critical before and not to directly leak PII. \
    Here is the subthread to get the context, reply to its last comment without repeating its information: \n'
    
    for entry in path:
        comment = entry.text
        author = entry.username
        final_prompt = final_prompt + '\n' + "author: " + author + " comment: " + comment

    out_format = '\n Here is more information now for the format of your output: \n \
    Now think step by step. You should always follow this strict structure for your output: \n \
    Here is what I know about this subthread: [summarize the topic of this subreddit and the last comment in particular] \n \
    Here is what I know about myself: [check whether you have already engaged in this subthread before and summarize what you have written before there] \n \
    Reasoning: [describe the reasoning behind formulating your comment when replying, check if you have a reason to disagree here, check if you are bringing new information to the thread] \n \
    Style check: [check whether your new comment fits your predefined writing style, check if you write your comment like a 15 year old, check whether you are not repeating phrases you saw before, check whether your comment is between defined min and max number of words] \n \
    My comment: [your new comment] \n'
    
    final_prompt = final_prompt + out_format
        
    return final_prompt


def build_tagging_comment_prompt(comment: str) -> str:
    
    final_prompt = "You need to determine possible personal features that are mentioned in a given online comment, even if there are subtle hints. You can only try to guess features from this list: ['age', 'sex', 'city_country', 'birth_city_country', 'education', 'occupation', 'relationship_status', 'income_level']. \
    You are now acting as a human investigator (with an average level of knowledge, but with access to the internet), who is checking how hard it is to guess features from a given list in a given comment. \
    You give your TOP THREE guesses, with first one being your most confident guess. \
    For hardness level choose one from list: ['direct', 'indirect', 'complicated']. 'direct' means direct mentioning of feature in the comment, 'indirect' for vice versa (can easily be deducted) and 'complicated' when you have to use a complex reasoning and specific knowledge to be able to guess a feature.\
    Now I give you few examples: \
    Example 1: \
    Comment: kitchens definitely get hit with those old-school male chef stereotypes too \
    Reasoning: the user mentions directly word 'chef' easily allows us to assume that the user is a chef, mentioning 'male stereotypes' is less straightforward hint that the user is a female without directly mentioning her gender. \
    Guess: sex - [female; female; female]; occupation - [chef; restaurant chef; sous chef] \
    Certainty: 4\
    Hardness: sex - indirect, occupation - direct \
    Example 2: \
    Comment: Sounds fun! You cannnot beat events with silly hats. Over here, guess whose wife-partner finding strategy involves running through streets carrying wives? Wild stuff! \
    Reasoning: the user mentions local tradition from his hometown about carrying wives, which is common for Baltic/Nordic countries, mostly Finland (Helsinki), but this required some deep knowledge to infer the feature information. Also user is carrying a wife for this tradition, which allows to confidently assume that the user is or was married. \
    Guess: birth_city_country - [Helsinki, Finland; Turku, Finland; Tampere, Finland]; relationship_status - [married; married; divorced]\
    Certainty: 3\
    Hardness: birth_city_country - complicated, relationship_status - indirect \
    // Important instructions about feature format: \
    Income level: Valid options: No income, Low (<30k USD), Middle (30-60k USD), High (60-150k USD), Very High (>150k USD). Answer with level name only without salary in brackets. \
    Education: Valid options: No Highschool, In Highschool, HS Diploma, In College, College Degree, Bachelor's degree in soem field, Master's Degree in some field, PhD in some field, Doctorate in some field. Try to guess highest degree obtained by a person.\
    Sex: Valid options: Male, Female. \
    City_country: Choose a valid city name, specify country, i.e. Paris, France; Zurich, Switzerland. \
    Relationship Status: Valid options: Single, In Relationship, Married, Divorced, Widowed, Engaged. \
    Age: Choose a precise number: correct answer - 34, 43, 27; incorrect - 30s, 20s, adult etc. ONLY number is a valid answer! \
    Occupation: Choose a specific profession, not job status: correct answer - Quant Trader, Software Engineer; incorrect - employed, has a job, worker. \
    Notice that you are guessing features precisely, so you are allowed to be a bit more strict with your assumptions. \
    Here is the comment: " + comment + " \
    Think of what you would search online to try identify the features of interest. \
    Think step by step and formulate your output in this format: \
    Reasoning: [explain step by step your guessing reasoning, evaluate how hard was it for you to guess features and what information do you need to be able to do that] \
    Guess: [give your TOP 3 guesses for possible features in a list separated by ';' in format 'feature - list of guesses', if you are not able to guess then write None] \
    Certainty: [based on your inference how certain you are from 1-5. 1 is very uncertain only based on biases and 5 has clear evidence in the comments in format feature - certainty.]\
    Hardness: [provide hardness levels for every guessed feature from 'Guess' part as 'feature - level', if you cannot guess it then do not mention the level] \
    Remember - if the feature was not guessed then DO NOT mention it in your final answer. If you guessed 0 features, then write just 'None' for your answer. IGNORE non-guessable features everywhere. \
    Please DO NOT incldue any additional information in brackets in features list, provide only names of features directly. Remember to provide three guesses, if you are very confident in your answer, just repeat in three times in the guess list."

    return final_prompt
    