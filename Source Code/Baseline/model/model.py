from os import lstat
from time import time
from torch import lstm, relu
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class MotionSenseModel(BaseModel):
    def __init__(self, time_slice:int, dim_latent:int, dim_lstm_hidden:int):
        super().__init__()
        self.num_classes = 6
        self.time_slice = time_slice
        self.dim_latent =dim_latent
        self.dim_lstm_hidden = dim_lstm_hidden

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv2_drop = nn.Dropout2d()
        
        assert time_slice%4==0

        self.dim_per_batch = (time_slice//4) * 3 * 64

        self.fc1 = nn.Linear(self.dim_per_batch, time_slice*dim_latent)

        self.lstm = nn.LSTM(input_size=dim_latent, hidden_size=dim_lstm_hidden, num_layers=1, batch_first=True)

        self.fc2 = nn.Linear(dim_lstm_hidden, 6)
        
    def forward(self, x):
        # Encoder
        print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)).view(-1, self.dim_per_batch)
        print(x.shape)
        x = F.relu(self.fc1(x)).view(-1, self.time_slice, self.dim_latent)#.unsqueeze(-1)

        #Decoder
        _, (hn, _) = self.lstm(x)

        x = hn.squeeze(0)

        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
        

class MotionSenseCNN(BaseModel):
    def __init__(self, time_slice:int, dim_latent:int, dim_lstm_hidden:int):
        super().__init__()
        self.num_classes = 6
        self.time_slice = time_slice
        self.dim_latent =dim_latent
        self.dim_lstm_hidden = dim_lstm_hidden

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        
        assert time_slice%4==0

        self.dim_per_batch = (time_slice//4) * 3 * 64

        self.fc1 = nn.Linear(self.dim_per_batch, time_slice*dim_latent)

        self.fc2 = nn.Linear(time_slice*dim_latent, 6)
        
    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        x = F.relu(F.max_pool2d(self.conv2(x), 2)).view(-1, self.dim_per_batch)
    
        x = F.relu(self.fc1(x))#.unsqueeze(-1)

        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class MotionSenseLSTM(BaseModel):
    def __init__(self, time_slice:int, dim_latent:int, dim_lstm_hidden:int):
        super().__init__()
        self.num_classes = 6
        self.time_slice = time_slice
        self.dim_latent =dim_latent
        self.dim_lstm_hidden = dim_lstm_hidden

        assert time_slice%4==0

        self.dim_per_batch = 144

        self.fc1 = nn.Linear(self.dim_per_batch, time_slice*dim_latent)

        self.lstm = nn.LSTM(input_size=dim_latent, hidden_size=dim_lstm_hidden, num_layers=1, batch_first=True)

        self.fc2 = nn.Linear(dim_lstm_hidden, 6)
        
    def forward(self, x):
        # Encoder

        x = F.relu(self.fc1(x.view(-1, 144))).view(-1, self.time_slice, self.dim_latent)#.unsqueeze(-1)

        #Decoder
        _, (hn, _) = self.lstm(x)

        x = hn.squeeze(0)

        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)