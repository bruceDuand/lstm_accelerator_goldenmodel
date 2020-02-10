import torch
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from constants import NUM_MFCC

class Args(object):
    def __init__(self):
        # environment set
        self.is_cuda = False
        self.seed = 0

        # conv layer params
        self.conv1_out_channels = 16
        self.conv2_out_channels = 8

        # LSTM layer params
        # self.num_memory_cts = 16
        self.num_memory_cts = 60
        self.input_size = 5
        self.sequence_length = 5
        self.batch_size = 1
        self.num_layers = 2

        # fc layer
        # self.fc_in_size = 80  # equals to 41*self.num_memory_cts
        # self.fc1_out_size = 64
        # self.fc2_out_size = 32
        self.fc_in_size = 600
        self.fc1_out_size = 256
        self.fc2_out_size = 128


args = Args()
args.is_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)


class GenderClassifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(GenderClassifier, self).__init__()
        self.args = args

        # input shape = (batch_size, input_channels, sequence_length)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=NUM_MFCC, out_channels=self.args.conv1_out_channels, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.args.conv1_out_channels, out_channels=self.args.conv2_out_channels, kernel_size=2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=16)
            nn.MaxPool1d(kernel_size=8)
        )

        # self.lstm = nn.LSTM(input_size=self.args.conv2_out_channels, hidden_size=args.num_memory_cts, batch_first=True)
        self.lstm = nn.LSTM(input_size=self.args.conv2_out_channels, hidden_size=self.args.num_memory_cts, batch_first=True)

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=self.args.fc_in_size, out_features=self.args.fc1_out_size),
            nn.Linear(in_features=self.args.fc1_out_size, out_features=self.args.fc2_out_size),
            nn.Linear(in_features=self.args.fc2_out_size, out_features=NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))
        c0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))

        cout = self.conv(x)
        lstm_out, _ = self.lstm(cout.transpose(1, 2), (h0, c0))
        fc_out = self.fc(lstm_out)

        return fc_out

    def get_conv1_out(self, x):
        cout = self.conv(x)
        return cout

    def get_lstm_out(self, x):
        h0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))
        c0 = torch.zeros((1, x.size(0), self.args.num_memory_cts))

        cout = self.conv(x)
        lstm_out, _ = self.lstm(c1out.transpose(1, 2), (h0, c0))
        return c1out, lstm_out