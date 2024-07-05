# SynthPAI configuration for generation of original dataset

Configuration .yaml files are located in two folders depending on the task:
- [`/configs/eval`](https://github.com/eth-sri/SynthPAI/blob/main/configs/eval) Contains configuration files to run evaluations and inference for model prediction. Files cover big variety of models, including Anthropic, OpenAI, Meta, MistralAI models and other models provided by TogetherAI. Files are sorted in folders for different model providers.
- [`/configs/eval/eval_labels.yaml`](https://github.com/eth-sri/SynthPAI/blob/main/configs/eval/eval_labels.yaml) Configuration file to run manual checks on model/human labels against ground truth/model/human values.
- [`/configs/thread/thread.yaml`](https://github.com/eth-sri/SynthPAI/blob/main/configs/thread/thread.yaml) Configuration file to run thread generation.

## Thread generation configuration

In [thread configuration file](https://github.com/eth-sri/SynthPAI/blob/main/configs/thread/thread.yaml) one is able to choose the LLM to generate thread topic (*author_bot*) and comments (*user_bot*) - recommended to use GPT-4 as in the paper. *Checker_bot* is used for prompting it the thread topic and synthetic profiles 1 by 1 to test if they would be interested in engaging in such topic online. The respective model parameters can also be changed in the file.
Task config parameters include the following:
1. `no_threads`: number of threads generated for attribute during one run; 1 used for generation of original dataset. 
2. `no_rounds`: number of engagements per user in the same thread; 2 used for generation of original dataset. 
3. `no_actions`: number of actions for user to take during one round in the same thread; 3 used for generation of original dataset. 
4. `no_max_comments`: maximum number of comments replying to the same comment; 3 used for generation of original dataset. 
5. `max_depth`: maximum depth of children subtree belonging to comment; 5 used for generation of original dataset. 
6. `no_profiles`: number of interested profiles in the topic to sample from profile pool; 40 used for generation of original dataset. 
7. `p_critic`: probability of user writing comments in a more disagreeing manner; 0.7 used for generation of original dataset; 
8. `p_short`: probability of a generated comment being shorter than maximum length (max_comment_len); 0.7 used for generation of original dataset. 
9. `min_comment_len`: minimum length (in words) of a generated comment; 5 used for generation of original dataset. 
10. `max_comment_len`: maximumlength(in words) of a generated comment when it is selected to be a short comment; 20 used for generation of original dataset. 
11. `no_sampled_comments`: number of comments to pick to choose from to answer; 10 used for generation of original dataset. 
12. `default_comment_prob`: start probability of user taking action "reply" in thread, decreases with every round; 0.7 used for generation of original dataset;

**Important**: If you updated synthetic profile dataset file, you need to regenerate writing styles for them using [this configuration file](https://github.com/eth-sri/SynthPAI/blob/main/configs/thread/generate_styles.yaml).

## Evaluation of labels configuration

In the [respective configuration file](https://github.com/eth-sri/SynthPAI/blob/main/configs/eval/eval_labels.yaml) one can specify which labels to evaluate against which to assert their quality. There are three options available: `gt` for ground truth labels from original synthetic profile dataset, `model` for LLM tags and `human` for human tags. There are three `decider` modes available for evaluation - `human` for completely manual check by hand; `model` for completely automated process (evaluation is done by `gen_model`); `model_human` for mixed evaluation - first model checks if the labels are close, if their accuracy is 0 then a user is asked to evaluate the label similarity themselves.

## LLM Evaluation configuration

The [`/configs/eval`](https://github.com/eth-sri/SynthPAI/blob/main/configs/eval) folder contains configuration files for both prediction and guessing evaluation setups for each model as used in the original paper.
One can choose on comments with which labels to predict/evaluate on - `profile_filter` parameter is designed to filter out tags (labels for specific features in comments) which have hardness and certainty levels lower than ones specified in the configuration file. Authors recommend to use `hardness: 1 certainty: 1` for prediction and then the suer can filter out comments for evaluation later. We recommend using `hardness: 1 certainty: 3` for evaluation to replicate original paper results as the labels with certainty level less than 3 are not reliable.
Prediction configs have `eval=False` set, while evaluation configs have `eval=True`.
Same as for label evaluation process, there are three modes of evaluation (`decider`) available - `human` for completely manual check by hand; `model` for completely automated process (evaluation is done by `gen_model`); `model_human` for mixed evaluation - first model checks if the labels are close, if their accuracy is 0 then a user is asked to evaluate the label similarity themselves.
Then the user can configure which labels to evaluate against (`label_type`) - `gt` for ground truth labels (from synthetic profiels dataset) and `human` for human crafted ones. We recommend using `human` to replicate results from the original paper as the goal is to compare LLMs against humans. For `label_type: human` there are two options available, configured in `human_label_type`: `original` for original human guesses/tags and `revised`, where the ones with accuracy 0 to ground truth were removed. We **strongly** recommend using `revised` type for this parameter.
