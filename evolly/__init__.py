__version__ = '0.1.0'

from .evolution import Evolution
from .model_builder import build_model
from .utils import compute_fitness, unpack_genotype, pack_genotype

from . import blocks

from .blocks.tensorflow.misc import get_flops_tf
from .blocks.torch.misc import GetFlops as GetFlopsTorch

from .analyze_runs import analyze_runs
from .visualize_run import visualize_run
from . import utils
