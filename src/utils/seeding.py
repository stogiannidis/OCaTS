# Standard library imports
import random  # For random number generation

# Third-party library imports
import numpy as np  # For numerical computations
import torch  # PyTorch deep learning library

#TPU-specific imports
flag = False
try:
    import torch_xla  # PyTorch XLA for TPU support
    import torch_xla.core.xla_model as xm  # XLA model support
    flag = True
except ImportError:
    pass


def set_seed(seed):
    """
    A function to set the seed for the random number generator.

    Parameters
    ----------
    seed
        The seed to set.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if flag:
        xm.set_rng_state(seed)
    # Additional libraries and frameworks, if needed