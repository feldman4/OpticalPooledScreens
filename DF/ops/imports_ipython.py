# plotting for notebook and code reloading 

from ops.imports import *
from ops.firesnake import Snake
from ops.process import Align

import IPython
IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
IPython.get_ipython().run_line_magic('autoreload', '2')


import matplotlib.pyplot as plt
import seaborn as sns

import tqdm.notebook
tqdn = tqdm.notebook.tqdm