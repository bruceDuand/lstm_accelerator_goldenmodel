import torch
import torch.nn as nn

import numpy as np

conv1_weights   = np.loadtxt("weights_data/conv_0_weights.txt")
print(conv1_weights.shape)