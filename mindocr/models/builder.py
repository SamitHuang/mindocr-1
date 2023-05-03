'''
build models
'''
import os
from typing import Union
from mindspore import load_checkpoint, load_param_into_net
from ._registry import model_entrypoint, list_models, is_model
from .base_model import BaseModel

__all__ = ['build_model']


def build_model(name_or_config: Union[str, dict], **kwargs):
    '''
    There are two ways to build a model.
        1. load a predefined model according the given model name.
        2. build the model according to the detailed configuration of the each module (transform, backbone, neck and head), for lower-level architecture customization.

    Args:
        name_or_config (Union[dict, str]): model name or config
            if it's a string, it should be a model name (which can be found by mindocr.list_models())
            if it's a dict, it should be an architecture configuration defining the backbone/neck/head components (e.g., parsed from yaml config).

        kwargs (dict): options
            if name_or_config is a model name, supported args in kwargs are:
                - pretrained (bool): if True, pretrained checkpoint will be downloaded and loaded into the network.
                - ckpt_load_path (str): path to checkpoint file. if a non-empty string given, the local checkpoint will loaded into the network.
            if name_or_config is an architecture definition dict, supported args are:
                - ckpt_load_path (str): path to checkpoint file.

    Return:
        nn.Cell

    Example:
    >>>  from mindocr.models import build_model
    >>>  net = build_model(cfg['model'])
    >>>  net = build_model(cfg['model'], ckpt_load_path='./r50_fpn_dbhead.ckpt') # build network and load checkpoint
    >>>  net = build_model('dbnet_resnet50', pretrained=True)

    '''
    if isinstance(name_or_config, str):
        # build model by specific model name
        model_name = name_or_config
        if is_model(model_name):
            create_fn = model_entrypoint(model_name)
            network = create_fn(**kwargs)
        else:
            raise ValueError(f'Invalid model name: {model_name}. Supported models are {list_models()}')

    elif isinstance(name_or_config, dict):
        #config_dict = name_or_config
        #pretrained_ckpt_path = config_dict.pop('pretrained_ckpt_path', None)
        # build model by given architecture config dict
        network = BaseModel(name_or_config)
    else:
        raise ValueError('Type error for config')

    # load checkpoint
    if 'ckpt_load_path' in kwargs:
        ckpt_path = kwargs['ckpt_load_path']
        if ckpt_path is not None:
            assert os.path.exists(ckpt_path), f'Failed to load checkpoint. {ckpt_path} NOT exist. \n Please check the path and set it in `eval-ckpt_load_path` or `model-pretrained_path` in the yaml config file '
            params = load_checkpoint(ckpt_path)
            load_param_into_net(network, params)
            print(f'INFO: Loaded checkpoint from {ckpt_path}')

    return network
