from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PBM
from pydantic import Extra, Field


class Task(Enum):
    THREAD = "THREAD"
    GENSTYLETHREAD = "GENSTYLETHREAD"
    EVAL = "EVAL"
    EVALLabels = "EVALLabels"


class ModelConfig(PBM):
    name: str = Field(description="Name of the model")
    tokenizer_name: Optional[str] = Field(
        None, description="Name of the tokenizer to use"
    )
    provider: str = Field(description="Provider of the model")
    dtype: str = Field(
        "float16", description="Data type of the model (only used for local models)"
    )
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers (Batch-size) to use for parallel generation"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the model upon generation",
    )
    model_template: str = Field(
        default="{prompt}",
        description="Template to use for the model (only used for local models)",
    )
    prompt_template: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the prompt"
    )
    submodels: List["ModelConfig"] = Field(
        default_factory=list, description="Submodels to use"
    )
    multi_selector: str = Field(
        default="majority", description="How to select the final answer"
    )

    def get_name(self) -> str:
        if self.name == "multi":
            return "multi" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.name == "chain":
            return "chain_" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.provider == "hf":
            return self.name.split("/")[-1]
        else:
            return self.name


class BasePromptConfig(PBM):
    # Contains prompt attributes which pertain to the header and footer of the prompt
    # These attributes are not used in the intermediate text (i.e. the meat of the prompt)
    modifies: List[str] = Field(
        default_factory=list,
        description="Whether this prompt config is used to modify existing prompts",
    )
    num_answers: int = Field(3, description="Number of answer given by the model")
    num_shots: int = Field(
        0, description="Number of shots to be presented to the model"
    )
    cot: bool = Field(False, description="Whether to use COT prompting")
    use_qa: bool = Field(
        False, description="Whether to present answer options to the model"
    )
    header: Optional[str] = Field(
        default=None,
        description="In case we want to set a specific header for the prompt.",
    )
    footer: Optional[str] = Field(
        default=None,
        description="In case we want to set a specific footer for the prompt.",
    )

    # Workaround to use this as a pure modifier as well
    def __init__(self, **data):
        super().__init__(**data)
        self.modifies = list(data.keys())

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["dryrun", "save_prompts", "header", "footer", "modifies"]:
                continue

            if "_" in attr:
                attr_short = attr.split("_")[1][:4]
            else:
                attr_short = attr[:4]

            file_path += f"{attr_short}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"
    

class EVALConfig(PBM):
    path: str = Field(
        ...,
        description="Path to the file",
    )
    paths: List[str] = Field(
        default_factory=list,
        description="Paths to the files for merging",
    )
    outpath: str = Field(
        ...,
        description="Path to write to for comment scoring",
    )
    eval: bool = Field(
        default="False",
        description="Whether to only evaluate the corresponding profiles.",
    )
    eval_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Settings for evaluation.",
    )
    decider: str = Field(
        default="model", description="Decider to use in case there's no match."
    )
    label_type: str = Field(
        default="gt", description="Which labels compare guesses to - gt for ground trurth (original labels); human for human guesses"
    )
    human_label_type: str = Field(
        default="gt", description="Which labels compare guesses to - revised for revised labels, original for original ones"
    )
    profile_filter: Dict[str, int] = Field(
        default_factory=dict, description="Filter profiles based on comment statistics."
    )
    max_prompts: Optional[int] = Field(
        default=None, description="Maximum number of prompts asked (int total)"
    )
    header: Optional[str] = Field(default=None, description="Prompt header to use")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    individual_prompts: bool = Field(
        False,
        description="Whether we want one prompt per attribute inferred or one for all.",
    )

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["path", "outpath"]:
                continue
            if attr == "profile_filter":
                file_path += (
                    str([f"{k}:{v}" for k, v in getattr(self, attr).items()]) + "_"
                )
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid

class EVALLabelsConfig(PBM):
    path: str = Field(
        ...,
        description="Path to the file",
    )
    paths: List[str] = Field(
        default_factory=list,
        description="Paths to the files for merging",
    )
    outpath: str = Field(
        ...,
        description="Path to write to for comment scoring",
    )
    eval: bool = Field(
        default="False",
        description="Whether to only evaluate the corresponding profiles.",
    )
    eval_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Settings for evaluation.",
    )
    decider: str = Field(
        default="model", description="Decider to use in case there's no match."
    )
    true_label_type: str = Field(
        default="gt", description="Which labels compare guesses to - gt for ground trurth (original labels); human for human guesses"
    )
    eval_label_type: str = Field(
        default="gt", description="Which labels take as true ones"
    )
    max_prompts: Optional[int] = Field(
        default=None, description="Maximum number of prompts asked (int total)"
    )
    header: Optional[str] = Field(default=None, description="Prompt header to use")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    individual_prompts: bool = Field(
        False,
        description="Whether we want one prompt per attribute inferred or one for all.",
    )

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["path", "outpath"]:
                continue
            if attr == "profile_filter":
                file_path += (
                    str([f"{k}:{v}" for k, v in getattr(self, attr).items()]) + "_"
                )
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid



class THREADConfig(PBM):
    
    no_threads: int = Field(
        default=1,
        description="Number of generations of thread.",
    )

    no_rounds: int = Field(
        default=1,
        description="Number of rounds for 1 thread.",
    )

    no_actions: int = Field(
        default=1,
        description="Number of actions a bot can take"
    )

    no_max_comments: int = Field(
        default=1,
        description="Number of max number of comments a comment can have"
    )

    max_depth: int = Field(
        default=3,
        description="Number of max no. of comment levels in subthread"
    )

    mode: str = Field(
        default=None,
        description="Mode for sampling comments: random N ('random') or top-N ('top')"
    )

    no_sampled_comments: int = Field(
        default=5,
        description="Number of sampled comments for choosing"
    )

    default_comment_prob: int = Field(
        default=7,
        description="Starting probability of commenting the post/comment, i.e. 7/10=0.7=70%"
    )

    no_profiles: int = Field(
        default=10,
        description="Number of profiles engaging"
    )

    p_critic: float = Field(
        default=0.3,
        description="percent of critic profiles out of all"
    )

    p_short: float = Field(
        default=0.3,
        description="probability of restricting comment length"
    )

    min_comment_len: int = Field(
        default=1,
        description="Min length of generated comment (in words)"
    )

    max_comment_len: int = Field(
        default=10,
        description="Max length of generated comment (in words)"
    )


    author_bot_system_prompt_path: str = Field(
        default="./data/thread/system_prompts/author_system_prompt.txt",
        description="Path to the file containing the author bot system prompt",
    )

    user_bot_system_prompt_path: str = Field(
        default="./data/thread/system_prompts/user_system_prompt.txt",
        description="Path to the file containing the user bot system prompt",
    )

    profile_checker_prompt_path: str = Field(
        default="./data/thread/system_prompts/profile_checker_prompt.txt",
        description="Path to the file containing the profile checking system prompt",
    )

    user_style_prompt_path: str = Field(
        default="./data/thread/system_prompts/user_style_system_prompt.txt",
        description="Path to the file containing the user writing style system prompt",
    )

    guess_feature: list = Field(
        default=["city_country"],
        description="The features on which to generate synthetic content"
    )

    user_bot_personalities_path: str = Field(
        default="./data/curious_bots/user_bot_profiles.json",
        description="Path to the json file that stores the dictionary of the personalities",
    )

    user_bot_personality: int = Field(
        default=None,
        description="If this argument is set to an integer included in the .json containing the personalities, \
            then only this personality will be executed, otherwise, the whole range of personalities is iterated through.",
    )

    author_bot: ModelConfig = Field(
        default=None, description="Author model used in generation"
    )

    user_bot: ModelConfig = Field(
        default=None, description="User model used in generation"
    )

    checker_bot: ModelConfig = Field(
        default=None, description="Checker model used in generation"
    )

    class Config:
        extra = Extra.forbid

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if "path" in str(attr):
                filename_attr = str(getattr(self, attr)).replace('/', '_').replace('.', '')
                file_path += f"{attr}={filename_attr}"
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"
    
    class Config:
        extra = Extra.forbid



class Config(PBM):
    # This is the outermost config containing subconfigs for each benchmark as well as
    # IO and logging configs. The default values are set to None so that they can be
    # overridden by the user
    output_dir: str = Field(
        default=None, description="Directory to store the results in"
    )
    seed: int = Field(default=42, description="Seed to use for reproducibility")
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    task_config: ( THREADConfig | EVALConfig | EVALLabelsConfig) = Field(
        default=None, description="Config for the task"
    )
    gen_model: ModelConfig = Field(
        default=None, description="Model to use for generation, ignored for CHAT task"
    )
    store: bool = Field(
        default=True, description="Whether to store the results in a file"
    )
    save_prompts: bool = Field(
        False, description="Whether to ouput the prompts in JSON format"
    )
    dryrun: bool = Field(
        False, description="Whether to just output the queries and not predict"
    )
    timeout: int = Field(
        0.5, description="Timeout in seconds between requests for API restrictions"
    )

    def get_out_path(self, file_name) -> str:
        path_prefix = "results" if self.output_dir is None else self.output_dir

        model_name = self.gen_model.get_name()
        file_path = (
            f"{path_prefix}/{self.task.value}/{model_name}/{self.seed}/{file_name}"
        )
        if self.task.value == "CHAT":
            investigator_bot_name = self.task_config.investigator_bot.get_name()
            user_bot_name = self.task_config.user_bot.get_name()
            file_path = f"{path_prefix}/{self.task.value}/{investigator_bot_name}-{user_bot_name}/{self.seed}/{self.task_config.guess_feature}/{file_name}"
        elif self.task.value == "CHAT_EVAL":
            file_path = '/'.join((self.task_config.chat_path_prefix).split('/')[:-1]) + '/' + file_name
        else:
            model_name = self.gen_model.get_name()
            file_path = (
                f"{path_prefix}/{self.task.value}/{model_name}/{self.seed}/{file_name}"
            )


        return file_path
    
