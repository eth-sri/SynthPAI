from src.configs import ModelConfig

from .chain_model import ChainModel
from .model import BaseModel
from .multi_model import MultiModel
from .open_ai import OpenAIGPT
from .hf_model import HFModel
from .gcp.gcp_model import GCPModel
from .together_model import TogetherModel
from .anthropic_model import AnthropicModel


def get_model(config: ModelConfig) -> BaseModel:
    if config.provider == "openai" or config.provider == "azure":
        return OpenAIGPT(config)
    elif config.provider == "hf":
        return HFModel(config)
    elif config.provider == "together":
        return TogetherModel(config)
    elif config.provider == "anthropic":
        return AnthropicModel(config)
    elif config.provider == "gcp":
        return GCPModel(config)
    elif config.provider == "loc":
        if config.name == "multi":
            models = []
            for sub_cfg in config.submodels:
                models.append(get_model(sub_cfg))

            return MultiModel(config, models)
        if config.name == "chain":
            models = []
            for sub_cfg in config.submodels:
                models.append(get_model(sub_cfg))

            return ChainModel(config, models)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
