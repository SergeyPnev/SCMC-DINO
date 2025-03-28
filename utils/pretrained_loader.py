import os
import torch
from pprint import pprint
from collections import OrderedDict


def state_dict_from_path(path):
    """
    Returns the state dict from a checkpoint path
    """
    path = os.path.abspath(path)
    mname = os.path.basename(path)
    if os.path.isfile(path):
        # print("Loading weights from \"{}\"".format(mname))
        return torch.load(path)["state_dict"]
    else:
        dirname = os.path.dirname(path)
        raise FileNotFoundError(
            'Model "{}" is not present in "{}"'.format(mname, dirname)
        )


def load_from_pretrained(model, path, strict=False, drop_fc=False):
    """
    Loads wights to the input model from a pretrained model path
    """
    pretrained_state = state_dict_from_path(path).state_dict()
    # print(pretrained_state.keys())

    if drop_fc:
        pretrained_state = OrderedDict(
            [
                (k, v)
                for k, v in pretrained_state.items()
                if not k.startswith(("fc", "aux_fc", "teacher_encoder", "student_encoder.dmso_fc"))
            ]
        )
    dif_keys = model.load_state_dict(pretrained_state, strict=strict)
    dif_keys = set([" : ".join(key.split(".")[:2]) for key in dif_keys.unexpected_keys])
    if dif_keys:
        print("\033[93mUnmatched pretrained modules\033[0m")
        pprint(dif_keys)
    else:
        print(f"\033[92mPretrained model loaded from {os.path.basename(path)}\033[0m")
