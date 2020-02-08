import sys
sys.path.append('../')

import os
import torch
import numpy as np

from torch.autograd import Variable
from GenderClassifier import GenderClassifier
from preprocessings import get_mfccs, load_from_pickle
from args import Args
from constants import NUM_MFCC, NUM_FRAMES

from fwdfunctions import sigmoid, softmax, get_conv_out, get_fc_out, get_lstm_out, get_pool_out

#=======================================================
# Load Model & weights extraction
#=======================================================
model = GenderClassifier()
model.load_state_dict(torch.load('../model_state_dict.pkl'))
model.eval()

layer_list = list(model.state_dict().keys())

conv1_weights   = model.state_dict()[layer_list[0]].detach().numpy()
conv1_bias      = model.state_dict()[layer_list[1]].detach().numpy()
conv2_weights   = model.state_dict()[layer_list[2]].detach().numpy()
conv2_bias      = model.state_dict()[layer_list[3]].detach().numpy()

lstm_weight_ih  = model.state_dict()[layer_list[4]].detach().numpy()
lstm_weight_hh  = model.state_dict()[layer_list[5]].detach().numpy()
lstm_bias_ih    = model.state_dict()[layer_list[6]].detach().numpy()
lstm_bias_hh    = model.state_dict()[layer_list[7]].detach().numpy()

fc1_weight  = model.state_dict()[layer_list[8]].detach().numpy()
fc1_bias    = model.state_dict()[layer_list[9]].detach().numpy()
fc2_weight  = model.state_dict()[layer_list[10]].detach().numpy()
fc2_bias    = model.state_dict()[layer_list[11]].detach().numpy()
fc3_weight  = model.state_dict()[layer_list[12]].detach().numpy()
fc3_bias    = model.state_dict()[layer_list[13]].detach().numpy()

args = Args()

#=======================================================
# Load Data
#=======================================================
x_valid_mfccs   = get_mfccs(pickle_file="X-valid-mfccs.pkl")
y_valid         = load_from_pickle(filename="y-valid.pkl")
x_valid_tensor  = Variable(torch.Tensor(x_valid_mfccs), requires_grad=False)
y_valid_tensor  = Variable(torch.LongTensor(y_valid), requires_grad=False)

x_test_mfccs   = get_mfccs(pickle_file="X-test-mfccs.pkl")
y_test         = load_from_pickle(filename="y-test.pkl")
x_test_tensor  = Variable(torch.Tensor(x_test_mfccs), requires_grad=False)
y_test_tensor  = Variable(torch.LongTensor(y_test), requires_grad=False)


x_data_tensor   = x_test_tensor
y_data_tensor   = y_test_tensor
x_data          = x_data_tensor.numpy()
y_data          = y_data_tensor.numpy()

#=======================================================
# Accuracy of valid data
#=======================================================
print("------------- Test Accuracy -------------")

outputs = model(x_data_tensor)
_, outputs_label = outputs.max(dim=1)
accuracy = int(sum(outputs_label == y_data_tensor))/len(y_data_tensor)
print("data size: {}, accuracy: {:.5f}%".format(len(y_data_tensor), 100*accuracy))

#=======================================================
# Forward Calculation: before quantization
#=======================================================
print("------------- Before Quantization -------------")

# forwarding data
def forward_data(x_data):
    conv1out    = get_conv_out(conv1_weights, conv1_bias, x_data, 4)
    conv2out    = get_conv_out(conv2_weights, conv2_bias, conv1out, 2)
    pooledout   = get_pool_out(conv2out, kernel_size=16)
    lstm_in     = np.transpose(pooledout, (0, 2, 1))
    lstmout     = get_lstm_out(lstm_in, lstm_weight_ih, lstm_weight_hh, lstm_bias_ih, lstm_bias_hh, hidden_size=16)
    fcin        = lstmout.reshape(lstmout.shape[0], -1)
    fc1out      = get_fc_out(fcin, fc1_weight, fc1_bias)
    fc2out      = get_fc_out(fc1out, fc2_weight, fc2_bias)
    fc3out      = get_fc_out(fc2out, fc3_weight, fc3_bias)

    return fc3out

fcout = forward_data(x_data)

classres = np.zeros((fcout.shape[0]))
for i in range(fcout.shape[0]):    
    classres[i] = np.argmax(softmax(fcout[i]))
accuracy = int(sum(classres == y_data))/len(y_data)
print("data size: {}, accuracy: {:.5f}%".format(len(y_data), 100*accuracy))
#=======================================================
# Forward Calculation: after quantization
#=======================================================
print("------------- After Quantization -------------")

name_list = []
for string in list(model.state_dict().keys()):
    name_list.append("../quantization/"+string.replace('.', '_')+".txt")

conv1_weights  = np.loadtxt(name_list[0])
conv1_weights  = conv1_weights.reshape(16, 40, 4)
conv1_bias     = np.loadtxt(name_list[1])
conv2_weights  = np.loadtxt(name_list[2])
conv2_weights  = conv2_weights.reshape(8, 16, 2)
conv2_bias     = np.loadtxt(name_list[3])

lstm_weight_ih = np.loadtxt(name_list[4])
lstm_weight_hh = np.loadtxt(name_list[5])
lstm_bias_ih   = np.loadtxt(name_list[6])
lstm_bias_hh   = np.loadtxt(name_list[7])

fc1_weight     = np.loadtxt(name_list[8])
fc1_bias       = np.loadtxt(name_list[9])
fc2_weight     = np.loadtxt(name_list[10])
fc2_bias       = np.loadtxt(name_list[11])
fc3_weight     = np.loadtxt(name_list[12])
fc3_bias       = np.loadtxt(name_list[13])
# fc3_bias       = np.zeros(2)

fcout  = forward_data(x_data)
classres = np.zeros((fcout.shape[0]))
for i in range(fcout.shape[0]):    
    classres[i] = np.argmax(softmax(fcout[i]))
accuracy = int(sum(classres == y_data))/len(y_data)
print("data size: {}, accuracy: {:.5f}%".format(len(y_data), 100*accuracy))