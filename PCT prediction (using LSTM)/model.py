import torch.nn as nn
import torch


# ......
class LSTM(nn.Module):

    # ...
    def __init__(self, input_dim=18, output_dim=6165, hidden_dim=512, num_layers=3, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 레이어들
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_dim//2, hidden_dim//4, batch_first=True, dropout=dropout)
        
        # Dense 레이어들
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim//4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(2048, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    # ...
    def forward(self, x):
        # 입력 shape: (batch_size, 18)
        # LSTM을 위해 (batch_size, seq_len=1, input_dim)로 변환
        x = x.unsqueeze(1)  # (batch_size, 1, 18)
        
        # LSTM 레이어들 통과
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)
        
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out3 = self.dropout(lstm_out3)
        
        # 마지막 timestep의 출력 사용
        lstm_out = lstm_out3[:, -1, :]  # (batch_size, hidden_dim//4)
        
        # Dense 레이어들 통과
        output = self.fc_layers(lstm_out)
        
        return output