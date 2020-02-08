import sys
sys.path.append('../')

import torch
import numpy as np

from torch.autograd import Variable
from GenderClassifier import GenderClassifier
from preprocessings import get_mfccs, load_from_pickle
from args import Args

# #=======================================================
# # Load Model & weights extraction
# #=======================================================
# model = GenderClassifier()
# model.load_state_dict(torch.load('../model_state_dict.pkl'))
# model.eval()

# layer_list = list(model.state_dict().keys())

# #=======================================================
# # Load Data
# #=======================================================
# x_valid_mfccs = get_mfccs(pickle_file="X-valid-mfccs.pkl")
# y_valid = load_from_pickle(filename="y-valid.pkl")
# x_valid_tensor = Variable(torch.Tensor(x_valid_mfccs), requires_grad=False)
# y_valid_tensor = Variable(torch.LongTensor(y_valid), requires_grad=False)

# args = Args()

# #=======================================================
# # Accuracy of valid data
# #=======================================================
# print("------------- Test Accuracy -------------")

# outputs = model(x_valid_tensor)
# _, outputs_label = outputs.max(dim=1)
# accuracy = int(sum(outputs_label == y_valid_tensor))/len(y_valid_tensor)
# print("data size: {}, accuracy: {:.3f}%".format(len(y_valid), 100*accuracy))


# #=======================================================
# # Forward Calculation
# #=======================================================
# print("------------- Before Quantization -------------")
# # print(layer_list)

# m_c1out = model.get_conv1_out(x_valid_tensor)
# m_cout, m_lstmout = model.get_lstm_out(x_valid_tensor)
# # print(c1out.detach().numpy()[1,:,0])

# conv1_weights   = model.state_dict()[layer_list[0]].detach().numpy()
# conv1_bias      = model.state_dict()[layer_list[1]].detach().numpy()
# conv2_weights   = model.state_dict()[layer_list[2]].detach().numpy()
# conv2_bias      = model.state_dict()[layer_list[3]].detach().numpy()

# lstm_weight_ih  = model.state_dict()[layer_list[4]].detach().numpy()
# lstm_weight_hh  = model.state_dict()[layer_list[5]].detach().numpy()
# lstm_bias_ih    = model.state_dict()[layer_list[6]].detach().numpy()
# lstm_bias_hh    = model.state_dict()[layer_list[7]].detach().numpy()

# x_data          = x_valid_tensor.numpy()

def get_conv_out(conv_weights, conv_bias, x_data, kernel_size):
    """
    conv_weights_dim    [NUM_out_channel, NUM_in_channel, kernel_size]
    conv_bias_dim       [NUM_out_channel,]
    x_data              [NUM_SAMPLES, NUM_MFCC, NUM_FRAMES]
    """
    res = np.zeros((x_data.shape[0], conv_weights.shape[0], x_data.shape[2]-kernel_size+1))
    for sidx in range(res.shape[0]):
        for i in range(res.shape[2]):
            convout = np.dot(conv_weights.reshape(conv_weights.shape[0], -1), x_data[sidx,:,i:i+kernel_size].flatten()) + conv_bias
            for k in range(convout.shape[0]):
                if convout[k] <= 0:
                    convout[k] = 0
            res[sidx,:,i] = convout
    return res

def get_pool_out(fwd_data, kernel_size):
    """
    fwd_data_dim    [NUM_SAMPLES, NUM_channel, NUM_FRAMES]
    pooled_data_dim [NUM_SAMPLES, NUM_channel, NUM_FRAMES/kernel_size]
    """
    res = np.zeros((fwd_data.shape[0], fwd_data.shape[1], int(fwd_data.shape[2]/kernel_size)))
    for i in range(int(fwd_data.shape[2]/kernel_size)):
        res[:,:,i] = np.amax(fwd_data[:,:,i*kernel_size:i*kernel_size+kernel_size-1],2)

    return res


def sigmoid(x):
    return 1./(1+np.exp(-x))
    bd = 2

    res = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        if x[idx] > bd:
            res[idx] = 1
        elif x[idx] < -bd:
            res[idx] = 0
        else:
            res[idx] = x[idx]/4+0.5
    return res

def tanh(x):
    return np.sinh(x)/np.cosh(x)
    bd = 1
    res = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        if x[idx] > bd:
            res[idx] = 1
        elif x[idx] < -bd:
            res[idx] = 0
        else:
            res[idx] = x[idx]
    return res


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def get_lstm_out(fwd_data, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
    """
    fwd_data_dim    [NUM_SAMPLES, pooled_NUM_FRAMES(seq_len), NUM_channel]
    outdata_dim     [NUM_SAMPLES, pooled_NUM_FRAMES(seq_len), hidden_size]
    (W_ii|W_if|W_ig|W_io)
    """
    res = np.zeros((fwd_data.shape[0], fwd_data.shape[1], hidden_size))
    input_size = fwd_data.shape[2]
    time_steps = fwd_data.shape[1]
    # print("res shape:", res.shape)
    h_state = np.zeros((hidden_size))
    cell_state = np.zeros((hidden_size))
    for sidx in range(res.shape[0]):
        
        for t in range(fwd_data.shape[1]):
            step_i = sigmoid(np.dot(weight_ih[0:hidden_size,:], fwd_data[sidx,t,:])+bias_ih[0:hidden_size]+np.dot(weight_hh[0:hidden_size,:], h_state) + bias_hh[0:hidden_size])

            step_f = sigmoid(np.dot(weight_ih[hidden_size:2*hidden_size,:], fwd_data[sidx,t,:])+bias_ih[hidden_size:2*hidden_size]+np.dot(weight_hh[hidden_size:2*hidden_size,:], h_state) + bias_hh[hidden_size:2*hidden_size])

            step_g = np.tanh(np.dot(weight_ih[2*hidden_size:3*hidden_size,:], fwd_data[sidx,t,:])+bias_ih[2*hidden_size:3*hidden_size]+np.dot(weight_hh[2*hidden_size:3*hidden_size,:], h_state) + bias_hh[2*hidden_size:3*hidden_size])
            
            step_o = sigmoid(np.dot(weight_ih[3*hidden_size:4*hidden_size,:], fwd_data[sidx,t,:])+bias_ih[3*hidden_size:4*hidden_size]+np.dot(weight_hh[3*hidden_size:4*hidden_size,:], h_state) + bias_hh[3*hidden_size:4*hidden_size])

            cell_state = step_g * step_i + cell_state * step_f
            h_state = step_o * np.tanh(cell_state)

            res[sidx,t,:] = h_state

    return res


def get_fc_out(fwd_data, fc_weight, fc_bias):
    fcout = np.dot(fwd_data, fc_weight.T) + fc_bias
    return fcout