import logging
import torch
import sys


def chunk(total, part):
    """Returns a list with elements `[part, ..., part, remainder]` which sum up
    to `total`. Returns `[part, ..., part]` if `remainder` is 0.

    input:
        total: number
        part: number

    output: list
    """

    num_parts = total // part
    remainder = total % part

    if remainder == 0:
        return [part] * num_parts
    else:
        return [part] * num_parts + [remainder]


def init(opt):
    # Numerical issues
    global epsilon
    epsilon = 1e-6

    # Random seed
    torch.manual_seed(opt.seed)

    # Default Tensor
    global cuda
    if torch.cuda.is_available() and opt.cuda:
        cuda = True
        torch.cuda.set_device(opt.device)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.enabled = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        cuda = False
        torch.set_default_tensor_type('torch.FloatTensor')
        opt.cuda = False

    # Logging
    global logger
    logger = logging.getLogger('CDAE')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Visdom
    if opt.visdom:
        try:
            import visdom
            global vis
            vis = visdom.Visdom()
        except ImportError:
            log_warning('Visdom server not available, disabling')
            opt.visdom = False
