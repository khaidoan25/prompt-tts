import os
import numpy as np
from collections import OrderedDict
from dataclasses import fields
from typing import Any, Tuple

from einops import rearrange
from functools import partial
import numpy as np
from typing import Any, Optional, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def get_cwd():
    while not os.getcwd().endswith("prompt-tts"):
        os.chdir("..")
        
    return os.getcwd()


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
    
    
def transform_to_code(normalized_code: torch.Tensor) -> np.ndarray:
    codes = (normalized_code / 2 + 0.5).clamp(0, 1)
    codes = codes.cpu().numpy()
    codes = (codes * 1023).round().astype('int')
    
    return codes


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)


def list_to_tensor(x_list: List[torch.Tensor], pattern="n b d -> b n d"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)  # (n b 1)
    m = rearrange(m, pattern)
    m = m.to(x)
    return x, m


def _samplewise_merge_tensors(*l, sep: Optional[torch.Tensor]=None):
    def _join(x: Tuple[torch.Tensor], sep: torch.Tensor):
        """
        Args:
            x: (k t d)
            sep: (d)
        """
        ret = x[0]
        for i in range(1, len(x)):
            ret = torch.cat((ret, sep[None], x[i]), dim=0)
        return ret
    if sep is None:
        cat = torch.cat
    else:
        cat = partial(_join, sep=sep)
    return [*map(cat, zip(*l))]