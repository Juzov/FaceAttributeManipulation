import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from generator import Generator

seed = 9
g_0 = Generator(seed)