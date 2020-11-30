# common imports for interactive work

import os
import re
from natsort import natsorted
from collections import OrderedDict, Counter, defaultdict
from functools import partial
from glob import glob
from itertools import product

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd
import skimage

from .annotate import annotate_labels, annotate_points, annotate_bases
from .annotate import GRMC
from .io import BLUE, GREEN, RED, MAGENTA, GRAY, CYAN, GLASBEY 
from .io import grid_view
from .filenames import name_file as name
from .filenames import rename_file as rename
from .filenames import parse_filename as parse
from .filenames import timestamp, file_frame
from .io import read_stack as read
from .io import save_stack as save

from .utils import (
    or_join, and_join,
    groupby_reduce_concat, groupby_histogram, replace_cols,
    pile, montage, make_tiles, trim, join_stacks,
    csv_frame,
)

from .plates import add_global_xy, add_row_col
from .pool_design import reverse_complement as rc
from . import in_situ