# plotting for notebook and code reloading 

from .imports import *
from .firesnake import Snake
from .process import Align

import IPython
IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
IPython.get_ipython().run_line_magic('autoreload', '2')


import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
tqdm.pandas() # adds GroupBy.progress_apply
