import numpy as np
import torch
from torch import nn
from utils import activation_func

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

        # self.activation = activation_func(activation)

        # LSTM handles prior actions (z_batch) with size (NL,4,208)
        # output is size 128
        self.lstm = nn.LSTM(208, 128, batch_first=True)
        # input to dense1 layer is (x_batch) with size (NL,559) # batch_first=true
        # should be 559 not 507 (compare with douzero and value in dmc/utils.py)
        # self.dense1 = nn.Linear(559 + 128, 512)
        # self.dense2 = nn.Linear(512, 512)
        # self.dense3 = nn.Linear(512, 512)
        # self.dense4 = nn.Linear(512, 512)
        # self.dense5 = nn.Linear(512, 512)
        # self.dense6 = nn.Linear(512, 1)

        """
        test
        activation func: relu, leaky relu, selu
        """

        self.dense1 = ResidualBlock(559 + 128,512, 'relu')
        self.dense2 = ResidualBlock(512,512, 'relu')
        self.dense3 = ResidualBlock(512,512, 'relu')
        self.dense4 = ResidualBlock(512,512, 'relu')
        self.dense5 = ResidualBlock(512,512, 'relu')
        self.dense6 = ResidualBlock(512,1)



    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)  # we don't care about hidden state h_n and cell state c_n at time n
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)

        residual = x


        x = self.dense1(x)
        # x = torch.relu(x)
        x = self.dense2(x)
        # x = torch.relu(x)
        x = self.dense3(x)
        # x = torch.relu(x)
        x = self.dense4(x)
        # x = torch.relu(x)
        x = self.dense5(x)
        # x = torch.relu(x)
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

    def resBlock(in, dense, ):
        x = torch.relu(x)



    def should_apply_shortcut(self):
        # TODO: implement
        # return self.in_
        pass


# basic res block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        self.in_features, self.out_features, self.activation = in_features, out_features, activation

        # self.blocks = nn.Identity()
        self.blocks = nn.Sequential(
            # conv(in_channels, out_channels, *args, **kwargs),
            # nn.BatchNorm2d(out_channels)
            nn.Linear(512, 512),
            activation_func(activation)
            )

        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_features != self.out_features