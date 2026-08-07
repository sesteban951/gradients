##
#
# Bundle Method
#
##

# directory imports
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("GRAD_ROOT_DIR") or os.path.dirname(HERE)
sys.path.append(ROOT)

# standard imports
import matplotlib.pyplot as plt

# custom imports
from src.function_examples import *
from utils.plotting import plot_landscape, plot_gradient_field, plot_surface
