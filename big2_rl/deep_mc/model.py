import numpy as np
import torch
from torch import nn


# actual model for one of the 4 players
class Big2Model(nn.Module):
    def __init__(self, device=0):
        super().__init__()
        if not device == "cpu":  # move device to right parameter
            device = 'cuda:' + str(device)
            self.to(torch.device(device))

        # LSTM handles prior actions (z_batch) with size (NL,4,208)
        # output is size 128
        self.lstm = nn.LSTM(208, 128, batch_first=True)
        # input to dense1 layer is (x_batch) with size (NL,559) # batch_first=true
        # should be 559 not 507 (compare with douzero and value in dmc/utils.py)
        self.dense1 = nn.Linear(559 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)  # we don't care about hidden state h_n and cell state c_n at time n
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:  # this is used during dmc/learn()
            return dict(values=x)
        else:
            # epsilon-greedy policy with 'flags.exp_epsilon' as parameter
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            # returns the action to take
            return dict(action=action)


class Big2ModelResNet(nn.Module):
    def __init__(self, device=0, activation='relu'):
        super().__init__()
        if not device == "cpu":  # move device to right parameter
            device = 'cuda:' + str(device)
            self.to(torch.device(device))

        # LSTM handles prior actions (z_batch) with size (NL,4,208)
        # output is size 128
        self.lstm = nn.LSTM(208, 128, batch_first=True)
        # input to dense1 layer is (x_batch) with size (NL,559) # batch_first=true
        # should be 559 not 507 (compare with douzero and value in dmc/utils.py)

        """
        residual blocks
        activation func: relu, leaky relu, selu
        """
        self.dense1 = nn.Linear(559 + 128, 512)
        self.dense2 = ResidualBlock(512,512, activation)
        self.dense3 = ResidualBlock(512,512, activation)
        self.dense4 = ResidualBlock(512,512, activation)
        self.dense5 = ResidualBlock(512,512, activation)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)  # we don't care about hidden state h_n and cell state c_n at time n
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        """
        residual blocks
        """
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)

        if return_value:  # this is used during dmc/learn()
            return dict(values=x)
        else:
            # epsilon-greedy policy with 'flags.exp_epsilon' as parameter
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            # returns the action to take
            return dict(action=action)


# basic res block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        self.in_features, self.out_features, self.activation = in_features, out_features, activation

        # self.blocks = nn.Identity()
        self.blocks = nn.Sequential(
            nn.Linear(in_features, out_features),
            )

        self.activate = activation_func(activation)

    def forward(self, x):
        residual = x.clone()
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


def activation_func(activation):
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3200, 512]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
    # This error can be resolved by setting inplace=False in nn.ReLU and nn.LeakyReLU in blocks.py.
    # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5

    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class Big2ModelConv(nn.Module):
    def __init__(self, device=0):
        super().__init__()
        if not device == "cpu":  # move device to right parameter
            device = 'cuda:' + str(device)
            self.to(torch.device(device))

        # LSTM handles prior actions (z_batch) with size (NL,4,208)
        self.conv_z_1 = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 57)),  # B * 4 * 208
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        # Squeeze(-1) B * 64 * 16
        self.conv_z_2 = torch.nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(5,), padding=2),  # 128 * 16
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )
        self.conv_z_3 = torch.nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(3,), padding=1),  # 256 * 8
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

        )
        self.conv_z_4 = torch.nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=(3,), padding=1),  # 512 * 4
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )


        # input to dense1 layer is (x_batch) with size (NL,559) # batch_first=true
        # should be 559 not 507 (compare with douzero and value in dmc/utils.py)
        self.dense1 = nn.Linear(559 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)  # we don't care about hidden state h_n and cell state c_n at time n
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:  # this is used during dmc/learn()
            return dict(values=x)
        else:
            # epsilon-greedy policy with 'flags.exp_epsilon' as parameter
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            # returns the action to take
            return dict(action=action)
