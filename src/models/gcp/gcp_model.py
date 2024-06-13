from src.configs import ModelConfig
from src.models import BaseModel
from src.prompts import Prompt
from src.utils.limiter import RateLimiter
from typing import List, Tuple, Iterator
from tqdm import tqdm
import time

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold


class GCPModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" in self.config.args.keys():
            self.config.args["max_output_tokens"] = self.config.args["max_tokens"]
            del self.config.args["max_tokens"]

        if "max_output_tokens" not in self.config.args.keys():
            self.config.args["max_output_tokens"] = 600

        # paste your GCP console project name below
        project_id = ""

        vertexai.init(project=project_id, location="us-central1")

    def predict(self, input: Prompt, **kwargs) -> str:
        # parameters = {
        #     "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        #     "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        #     "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        #     "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        # }

        if input.system_prompt is not None:
            system_prompt = input.system_prompt
        else:
            system_prompt = "You are an expert investigator and detective with years of experience in online profiling and text analysis."
        self.model = GenerativeModel(model_name=self.config.name, system_instruction=[system_prompt])

        input_text = self.apply_model_template(input.get_prompt())
        try:
            response = self.model.generate_content(
                contents=input_text,
                # Optional:
                generation_config=GenerationConfig(
                    temperature=self.config.args["temperature"],
                    candidate_count=1,
                    max_output_tokens=self.config.args["max_output_tokens"],
                    stop_sequences=["STOP!"],
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            return response.text
        except Exception as e: 
            print(e)
            return ""

    def predict_string(self, input: str, **kwargs) -> str:
        response = self.model.predict(input, **self.config.args)  # type: ignore

        return response.text

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        rl = RateLimiter(9, 30)

        ids_to_do = list(range(len(inputs)))

        while len(ids_to_do) > 0:
            for id in tqdm(ids_to_do):
                if not rl.record():
                    print("Rate limit exceeded, sleeping for 10 seconds")
                    time.sleep(60)
                    continue

                orig = inputs[id]
                answer = self.predict(inputs[id])

                yield (orig, answer)
                ids_to_do.remove(id)
