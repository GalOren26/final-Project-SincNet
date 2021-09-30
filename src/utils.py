import typing as tp
from types import SimpleNamespace

import torch
import yaml

from model.model import SincNet


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


def compute_chunk_info(params: NestedNamespace) -> tp.Tuple[int, int]:
    chunk_len = int(params.sample_rate * params.chunk_len_ratio)
    chunk_shift = int(params.sample_rate * params.chunk_shift_ratio)
    return chunk_len, chunk_shift


def get_params(path_to_cfg: str):
    with open(path_to_cfg) as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
        params = NestedNamespace(params)
    return params


def load_model(params, args, chunk_len):
    sinc_net = SincNet(chunk_len, params.data.timit.n_classes, args.model_type)
    checkpoint = torch.load(args.pretrained_model, map_location=torch.device(params.device))
    sinc_net.load_state_dict(checkpoint['model_state_dict'])
    sinc_net = sinc_net.to(params.device)
    sinc_net.eval()
    return sinc_net