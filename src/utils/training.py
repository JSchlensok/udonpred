import yaml
from pathlib import Path

import torch

from .constants import embedding_dimensions, model_classes
from src.models import FNN, CNN

def load_model(config_file: Path, checkpoint_file: Path, map_location: str | torch.device = "cpu") -> torch.nn.Module:
    model_config = yaml.safe_load(config_file.open())
    model_type = model_config["model"]["type"]
    model_params = model_config["model"]["params"]
    embedding_type = model_config["embedding_type"]
    embedding_dim = embedding_dimensions[embedding_type]
    model = model_classes[model_type](n_features=embedding_dim, **model_params)
    model.load_state_dict(torch.load(checkpoint_file, map_location=map_location))
    return model

def freeze_layers(model: torch.nn.Module, layers_to_freeze: int) -> torch.nn.Module:
    num_frozen = 0
    while num_frozen < layers_to_freeze:
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                layer.requires_grad = False
                num_frozen += 1
    return model
