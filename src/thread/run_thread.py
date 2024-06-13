import json
import os
import glob
import re
import random
import html
import numpy as np
from scipy.special import softmax
from src.configs import Config
from src.models.model_factory import get_model
from src.prompts import Prompt, Conversation
from .eval_thread import eval_guess
from .user_bot_system_prompt_builder import build_user_bot_system_prompt, build_author_bot_system_prompt, build_checker_bot_system_prompt, build_comment_path_prompt, build_writing_style_prompt, build_tagging_comment_prompt

class Node:
    def __init__(self, author, profile, username, text, parent):
        self.author = author
        self.profile = profile
        self.username = username
        self.text = text
        self.children = []
        self.parent = parent
        self.scores = {}
        self.guesses = []


class RedditThread:
    def __init__(self): # initiate by creating empty thread node w/o post topic
        
        self.root = Node(None, None, None, None, None)
        self.comments = []
        self.critics = []
    
    def score_path(self, pers, path, node=None, score=0.0):
                    # pass target person, comment, score+level init at 0
        
        if node.parent.author == None:
            score += 2.0
            # adding 1.5(or more) > 1 to motivate replying to main post, for nested subthreads will be penalized enough by depth; proposed values: 1.5 (min), 2.0 or 2.5 (max)
            path.append(node)
            return score, path
        
        if node.author == pers:
            score += 5.0    # proposed values: 3.0 (min), 5.0 (recommended) or 10.0 (max)
        else:
            score += 1.0
        path.append(node)

        if node.children == []:
                if node.author == pers:
                    score = 0.0

        if node.author == pers:
            if node.children == []:
                score = 0.0
            else:
                score = self.score_path(pers, path, node.parent, score)[0]
        else:
            score = self.score_path(pers, path, node.parent, score)[0]

        return score, path
    
    def get_path(self, path, comment):

        if path == None:
            path = []

        path.append(comment)

        if comment.parent.text != None:
            self.get_path(path, comment.parent)

        return path
    
    def choose_comment(self, pers, cfg):

        # mode: n random or top-n comments

        mc = cfg.task_config.no_max_comments
        n = cfg.task_config.no_sampled_comments
        mode = cfg.task_config.mode
        max_depth = cfg.task_config.max_depth

        # when choosing comment retrieve score per person

        if mode == "random": # N random sampling

            if len(self.comments) > n:
                n_random_comments = random.sample(self.comments, n)
            else:
                n_random_comments = self.comments
            
            comments_scored = []
            comments = []
            scores = []
            
            for i in range(len(n_random_comments)):
                comment = n_random_comments[i]
                final_score = comment.scores[pers]
                # exclude comments made by author or with score = 0.0 -> additional check just in case
                if comment.author != pers:
                    if final_score != 0.0:
                        scores.append(final_score)
                        comments.append(comment)
                        comments_scored.append((comment, final_score))
            
            # normalize scores using scipy softmax
            norm_scores = softmax(scores)
            
            comments_norm_scored = []
            for i in range(len(comments_scored)):
                comments_norm_scored.append((comments[i], scores[i], norm_scores[i]))

            i = 0
            while i < 1:
                target_comment = np.random.choice([comment for comment, _, _ in comments_norm_scored], 
                        replace=False, p=norm_scores)
                parent_node = target_comment
                target_path = self.get_path(None, parent_node)
                if len(target_path) <= max_depth:
                    if len(target_comment.children) < mc:
                        parent_node = target_comment
                        i += 1

        if mode == "top": # top-N sampling
            comments_scored = []
            comments = []
            scores = []
            
            for i in range(len(self.comments)):
                comment = self.comments[i]
                final_score = comment.scores[pers]
                comment_path = self.get_path(None, comment)
                # exclude comments made by author or with score = 0.0 -> additional check just in case
                if comment.author != pers:
                    if final_score != 0.0:
                        if len(comment_path) <= max_depth:  # filter out comments at depth from root >= max_depth
                            if len(comment.children) < mc:  # filter out comments with no. children > max no. children
                                comments_scored.append((comment, final_score))  
            
            if len(comments_scored) < n:
                comments_scored = []
                comments = []
                scores = []
                
                for i in range(len(self.comments)):
                    comment = self.comments[i]
                    final_score = comment.scores[pers]
                    # exclude comments made by author or with score = 0.0 -> additional check just in case
                    if comment.author != pers:
                        if final_score != 0.0:
                            comments_scored.append((comment, final_score))
                
            
            # sort and choose top-N comments by score
            
            comments_sorted = sorted(comments_scored, key=lambda x: x[1], reverse=True)
            if len(comments_sorted) > n:
                comments_sorted = comments_sorted[:n]
                
            for comment, score in comments_sorted:
                comments.append(comment)
                scores.append(score)
                
            # normalize scores using scipy softmax
            norm_scores = softmax(scores)

            i = 0
            while i < 1: # sample indices, not comments
                target_comment = np.random.choice(comments, replace=False, p=norm_scores) # too slow
                parent_node = target_comment
                if target_comment.parent.author != None:    # check if comment is root (should not restrict no. children for thread topic)
                    parent_node = target_comment
                    i += 1
                else: 
                    i += 1

        # set the score of chosen comment for chosen person to 0 to avoid commenting it again       
        parent_node.scores[pers] = 0.0

        return parent_node

    def choose_profiles(self, bot, profile_checker_prompt, user_bot_personalities, reddit_topic, no_profiles):
                        # pass profiles dict to sample from
        
        profile_count = 1 # iterate until reach cfg.no_profiles
        keys = list()
        for pers, pers_profile in user_bot_personalities.items():

            checker_bot_system_prompt = build_checker_bot_system_prompt(
                    prompt_skeleton=profile_checker_prompt,
                    profile=pers_profile,
                    topic=reddit_topic
                )
            conv_checker = Conversation(
                            system_prompt=checker_bot_system_prompt,
                            prompts=[]
                        )
            answer = bot.continue_conversation(conv_checker)
            if bool(re.search("[yesYES]", answer)): # added regex checker to check if user is "interested"
                keys.append(pers) # add profile

            profile_count += 1
            if profile_count > no_profiles:
                break
        
        sampled_profiles = {key: user_bot_personalities[key] for key in keys}

        return sampled_profiles
    

    def get_styles(self, bot, profiles, user_style_prompt):
                    # sample writing styles before running thread using GPT
        styles = {}
        
        for pers, profile in profiles.items():

            style_bot_system_prompt = build_writing_style_prompt(
                    prompt_skeleton=user_style_prompt,
                    profile=profile
                )
            conv_style = Conversation(
                            system_prompt=style_bot_system_prompt,
                            prompts=[]
                        )
            
            style = bot.continue_conversation(conv_style)
            
            styles[pers] = style

        return styles
        

    def create_thread(self, bot, pers, profile, author_sys_prompt, guess_feature, bot_0): 
                     # pass author, profile, author system prompt, feature to guess

        profile_skeleton = profile
        user_name = profile["username"]
        
        # fix guess feature string to pass to post builder prompt to fix ambiguity
        if guess_feature == "city_country":
            guess_feature = "current place of living (city and country)"
        if guess_feature == "birth_city_country":
            guess_feature = "place of birth (city and country)"
        if guess_feature == "relationship_status":
            guess_feature = "relationship status"
        if guess_feature == "income_level":
            guess_feature = "level of income" 
        if guess_feature == "sex":
            guess_feature = "gender" 
        if guess_feature == "occupation":
            guess_feature = "profession/current job"     
        if guess_feature == "education":
            guess_feature = "degree and field of studies"

        profile_skeleton["guess_feature"] = guess_feature

        author_bot_system_prompt = build_author_bot_system_prompt(
                    prompt_skeleton=author_sys_prompt,
                    profile=profile_skeleton
                )
        conv_author = Conversation(
                        system_prompt=author_bot_system_prompt,
                        prompts=[]
                    )
        self.root.author = pers
        self.root.profile = profile
        self.root.username = user_name
        self.root.text = bot.continue_conversation(conv_author)

        # now we label the thread topic: prompt GPT-4 to try guess which features are leaked/can be inferred from text (use checker bot model for temp=0)

        conv_tag = Conversation(
                        system_prompt="",
                        prompts=[]
                    )
        
        tag_prompt = build_tagging_comment_prompt(
                comment=self.root.text
            )
        
        conv_tag.add_prompt(Prompt(role='user', intermediate=tag_prompt))
        guessed_output = bot_0.continue_conversation(conv_tag)

        # extract guesses

        features = ['age', 'sex', 'city_country', 'birth_city_country', 'education', 'occupation', 'relationship_status', 'income', 'income_level']
        result = []

        for feature in features:
            guess_match = re.search(f'{feature} - \[(.*?)\]', guessed_output)
            hrd = re.findall(r'Hardness: (.+)', guessed_output)[0]
            crt = re.findall(r'Certainty: (.+)', guessed_output)[0]
            hardness_match = re.search(rf'{feature} - (\w+)', hrd)

            if guess_match and hardness_match:
                guesses = guess_match.group(1).split('; ')
                if 'None' not in guesses and guesses != None:
                    certainty = re.search(f'{feature} - (\d)', crt)
                    if feature == 'age':    # additional check for age (should be in integer format)
                        if guesses != None:
                            int_guesses = [int(re.search(r'\d+', guess).group()) for guess in guesses]
                        if int_guesses != None:
                            guesses = int_guesses
                        else:
                            guesses = ['None']              
                    model_eval = eval_guess(feature, guesses, profile, bot_0)
                    acc = 0.0
                    if model_eval != None:
                        acc = np.mean(model_eval)
                    print("Acc for feature ", feature, ":",  acc)
                    if acc > 0.0:
                        result.append({
                            'feature': feature,
                            'guesses': guesses,
                            'hardness': hardness_match.group(1),
                            'certainty': int(certainty.group(1)),
                            'model_eval': model_eval
                        })
                        print('Saving guess: ', result)  

        self.root.guesses = result

        self.root.parent = Node(None, None, None, None, None)  
        self.comments.append(self.root)

    def add_comment(self, bot, pers, profile, user_sys_prompt, min_comment_len, max_comment_len, sampled_profiles, cfg, bot_0): 
                    # pass user, profile, user system prompt, length range of comment
        
        # calculate tree path score calculation for sampled comments and build probabilistic sampling

        parent_node = self.choose_comment(pers, cfg) # get parent node for new comment being generated
        
        profile_skeleton = profile
        writing_style = profile["style"]
        user_name = profile["username"]
        p_short = cfg.task_config.p_short

        ct = False
        for i in range(len(self.critics)):
            if self.critics[i] == pers:
                ct = True # motivate negative sentiment if pers is in critics list
                break

        choices = ["short", "long"]
        action = random.choices(choices, weights=[p_short*10, (1-p_short)*10])[0]

        if action == "long":
            min_comment_len = 10
            max_comment_len = 50 
            # even though we prompt model to generate text between these lengths, the model does not always follow this instruction
        
        pers_user_bot_system_prompt = build_user_bot_system_prompt(
                prompt_skeleton=user_sys_prompt,
                profile=profile_skeleton,
                writing_style=writing_style,
                min_comment_len = min_comment_len,
                max_comment_len = max_comment_len,
                ct = ct
            )
        
        conv_user = Conversation(
                        system_prompt=pers_user_bot_system_prompt,
                        prompts=[]
                    )
        
        orig_comment_path = self.get_path(None, parent_node)

        comment_prompt = build_comment_path_prompt(
                path=orig_comment_path,
                pers=user_name
            )
        
        conv_user.add_prompt(Prompt(role='user', intermediate=comment_prompt))
        comment_text = bot.continue_conversation(conv_user)

        # Adding regex checks for special cases of model writing "my comment" in different forms due to high temperature            
        comment_text = comment_text.split('My comment:')[-1].replace("\n"," ") # retrieve generated comment
        comment_text = comment_text.split('My Comment:')[-1].replace("\n"," ") # uppercase check (rare occasion)
        comment_text = comment_text.split('My-comment:')[-1].replace("\n"," ") # dash check (very rare occasion)
        comment_text = comment_text.split('My_comment:')[-1].replace("\n"," ") # underscore check (very rare occasion)
        comment_text = comment_text.split('My new comment:')[-1].replace("\n"," ") # special case check (very rare occasion)
        comment = Node(pers, profile, user_name, comment_text, parent_node)
        parent_node.children.append(comment)
       
        # calculate score for new path for all people involved in thread
        for pers, _ in sampled_profiles.items():
            path_empty = []
            score, path = self.score_path(pers, path_empty, comment, score = 0.0)
            final_score = float(score / float(len(path)))
            comment.scores[pers] = final_score

        # now we label the comment: prompt GPT-4 to try guess which features are leaked/can be inferred from new comment (use checker bot model for temp=0)

        conv_tag = Conversation(
                        system_prompt="",
                        prompts=[]
                    )
        
        tag_prompt = build_tagging_comment_prompt(
                comment=comment_text
            )
        
        conv_tag.add_prompt(Prompt(role='user', intermediate=tag_prompt))
        guessed_output = bot_0.continue_conversation(conv_tag)
        print(guessed_output)

        # extract guesses

        features = ['age', 'sex', 'city_country', 'birth_city_country', 'education', 'occupation', 'relationship_status', 'income', 'income_level']
        result = []

        for feature in features:
            guess_match = re.search(f'{feature} - \[(.*?)\]', guessed_output)
            hrd = re.findall(r'Hardness: (.+)', guessed_output)
            if len(hrd) != 0:
                hrd = hrd[0]
            else:
                hrd = None
            crt = re.findall(r'Certainty: (.+)', guessed_output)[0]
            hardness_match = re.search(rf'{feature} - (\w+)', hrd)

            if guess_match and hardness_match:
                guesses = guess_match.group(1).split('; ')
                if 'None' not in guesses:
                    certainty = re.search(f'{feature} - (\d)', crt)
                    if feature == 'age':    # additional check for age (should be in integer format)
                        if guesses != None:
                            int_guesses = [int(re.search(r'\d+', guess).group()) for guess in guesses]
                            if int_guesses != None:
                                guesses = int_guesses
                            else:
                                guesses = ['None']
                        else:
                            guesses = ['None']
                    model_eval = eval_guess(feature, guesses, profile, bot_0)
                    acc = 0.0
                    if model_eval != None:
                        acc = np.mean(model_eval)
                    print("Acc for feature ", feature, ":",  acc)
                    if acc > 0.0:
                        result.append({
                            'feature': feature,
                            'guesses': guesses,
                            'hardness': hardness_match.group(1),
                            'certainty': int(certainty.group(1)),
                            'model_eval': model_eval
                        })
                        print('Saving guess: ', result)   
                        comment.guesses = result

        self.comments.append(comment)

def print_thread(node, depth=0):
    
    indent = "    " * depth
    comment = node['text']
    author = node['username']
    print('\n')
    print(f"{indent} {author}: {comment}")
    
    for child in node['children']:
        print_thread(child, depth + 1)

def node_to_dict(node):
    
    return {
        'author': node.author,
        'profile': node.profile,
        'username': node.username,
        'guesses': node.guesses,
        'text': node.text,
        'children': [node_to_dict(child) for child in node.children]
    }

def generate_new_filename(feature):
    # helper function to check directory of generated threads to avoid rewriting files
    directory = f'./data/thread/generated_threads/json_threads/{feature}/'
    pattern = f'thread_{feature}_*.json' 
    files = glob.glob(os.path.join(directory, pattern))
    
    highest_number = 0   
    number_extractor = re.compile(rf'thread_{feature}_(\d+)\.json')
    
    for file in files:
        filename = os.path.basename(file)        
        match = number_extractor.match(filename)
        if match:
            number = int(match.group(1))
            if number > highest_number:
                highest_number = number
    
    new_number = highest_number + 1
    new_filename = f'./data/thread/generated_threads/json_threads/{feature}/thread_{feature}_{new_number}.json'
    
    return new_filename

def save_thread_as_json(thread, filename):
    import json

    thread_dict = node_to_dict(thread.root)
    
    with open(filename, 'w') as f:
        json.dump(thread_dict, f)

def json_to_html(data, indent=0):
    # convert generated thread to HTML page for better readibility
    html_data = '<div style="margin-left: {}px; padding: 8px; background: #333; border-left: 5px solid rgba(255, 255, 255, 0.05); color: #fff">'.format(indent)
    html_data += '<p style="margin-top: 0; margin-bottom: 1px; font-family: Verdana; font-size: 12px;"><b>{}</b></p>'.format(html.escape(data['username']))
    html_data += '<p style="margin-top: 0; margin-bottom: 10px; font-family: Helvetica; font-size: 16px;">{}</p>'.format(html.escape(data['text']))
    for child in data.get('children', []):
        html_data += '<div style="margin-top: 10px; padding-left: 10px;">'
        html_data += json_to_html(child, indent + 10)
        html_data += '</div>'
    html_data += '</div>'

    return html_data

def save_as_html(filename, html_data):
    with open(filename, 'w') as f:
        f.write('<html><body>{}</body></html>'.format(html_data))

def run_thread(cfg: Config) -> None:

    author = get_model(cfg.task_config.author_bot)
    user = get_model(cfg.task_config.user_bot)
    checker = get_model(cfg.task_config.checker_bot)

    # load the system prompts
    with open(cfg.task_config.author_bot_system_prompt_path, 'r', encoding="utf-8", errors='ignore') as f:
        author_bot_system_prompt = f.read()
    with open(cfg.task_config.user_bot_system_prompt_path, 'r', encoding="utf-8", errors='ignore') as f:
        user_bot_system_prompt_skeleton = f.read()
    with open(cfg.task_config.profile_checker_prompt_path, 'r', encoding="utf-8", errors='ignore') as f:
        profile_checker_prompt = f.read()

    # load the personalities for the user bot
    with open(cfg.task_config.user_bot_personalities_path, 'r', encoding="utf-8", errors='ignore') as f:
        user_bot_personalities = json.load(f)

    nc = cfg.task_config.no_actions
    no_threads = cfg.task_config.no_threads
    guess_features = cfg.task_config.guess_feature
    min_comment_len = cfg.task_config.min_comment_len
    max_comment_len = cfg.task_config.max_comment_len
    default_comment_prob = cfg.task_config.default_comment_prob
    no_profiles = cfg.task_config.no_profiles
    no_rounds = cfg.task_config.no_rounds
    p_critic = cfg.task_config.p_critic

    
    for feature in guess_features:

        thread_names = []

        # generate number of threads as specified in config
        for t in range(no_threads):
            
            thread = RedditThread()
            
            # sample n profiles using GPT based on personal interest
            # shuffle the profiles (for now) after creating new thread
            keys = list(user_bot_personalities.keys())
            random.shuffle(keys)
            thread_author = keys[0] # use first profile from shuffled dict - technically random sampling
            thread_profile = user_bot_personalities[thread_author]
            keys.remove(thread_author)
            user_bot_personalities = {key: user_bot_personalities[key] for key in keys}
            

            thread.create_thread(author, thread_author, thread_profile, author_bot_system_prompt, feature, checker)
            reddit_topic = thread.root.text
            sampled_profiles = thread.choose_profiles(checker, profile_checker_prompt, user_bot_personalities, reddit_topic, no_profiles)    
            sampled_profiles = dict((k, user_bot_personalities[k]) for k in sampled_profiles)
            print(f'\n\n No. of engaged profiles: for {feature} thread no. {t+1}: ', len(list(sampled_profiles.keys())))
            
            # score root for all sampled profiles
            for pers, _ in sampled_profiles.items():
                path_empty = []
                score, path = thread.score_path(pers, path_empty, thread.root, score = 0.0)
                final_score = float(score / float(len(path)))
                thread.root.scores[pers] = final_score

            # iterate over the personalities and simualate the thread n=no_rounds times
            for n in range(no_rounds):

                # sample p% of critisizing profiles before start of round
                sampled_keys = list(sampled_profiles.keys())
                thread.critics = np.random.choice(sampled_keys, int(no_profiles*p_critic))

                for pers, pers_profile in sampled_profiles.items():
                    try:
                        if thread.root.text == None: # by default now topic will exist before iterating over sampled profiles
                            # if topic does not exist a profile can generate a subreddit post or abstain
                            choices = ["post", "abstain"]
                            action = random.choices(choices, weights=[7, 3])[0]
                            
                            if action == "post":
                                thread.create_thread(author, pers, pers_profile, author_bot_system_prompt, feature, checker)

                        else:
                            # next profiles are given choices: comment or abstain
                            # 1 bot samples random number of actions to perform
                            choices = ["comment", "abstain"]
                            nc_list = list(range(1, nc + 1))
                            no_choices = random.choice(nc_list)

                            default_abstain_prob = 10 - default_comment_prob # = 1-0.7 = 30%

                            for i in range(no_choices):
                                if n < default_comment_prob:
                                    prob_comment = default_comment_prob - n
                                    prob_abstain = default_abstain_prob + n
                                else: # if too many rounds (unlikely), just set comment prob very low
                                    prob_comment = 1
                                    prob_abstain = 9
                                action = random.choices(choices, weights=[prob_comment, prob_abstain])[0] # reduce probability of commenting after every round

                                if action == "comment":
                                    thread.add_comment(user, pers, pers_profile, user_bot_system_prompt_skeleton,
                                                                min_comment_len, max_comment_len, sampled_profiles, cfg, checker)

                    except Exception as e:
                        e = e
                        print('\n')
                        print(e)

                # after round exclude root from comments to avoid replying twice
                if n == 0:
                    thread.comments.remove(thread.comments[0])


            # count number of total and tagged comments in newly created thread
            print('\n Total comments in thread: ', len(thread.comments))
            tagged = 0
            for comment in thread.comments:
                if len(comment.guesses) > 0:
                    tagged += 1
            print('Labeled comments: ', tagged)
            
            # save the thread
            thread_filename = generate_new_filename(feature)
            thread_names.append(thread_filename)             
            save_thread_as_json(thread, thread_filename)

            # save in HTML format
            with open(thread_filename, 'r') as f:
                data = json.load(f)
            html_data = json_to_html(data)
            html_filename = thread_filename.replace("json", "html")
            save_as_html(html_filename, html_data)
            
        # print the thread
        for t in range(no_threads):
            print('\n')
            print(f'##### THREAD no. {t+1} for feature {feature} #####')
            filename = thread_names[t]
            with open(filename, 'r', encoding="utf-8", errors='ignore') as f:
                thread_data = json.load(f)
            print_thread(thread_data)
