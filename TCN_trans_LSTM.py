import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
SEQ_LEN = 96  # 输入序列长度
SHORT_TERM_PRED = 96  # 预测短序列长度
LONG_TERM_PRED = 240  # 预测长序列长度
BATCH_SIZE = 256
EPOCHS = 150
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 限制 TensorFlow 使用 GPU 1, 2, 3
# device = tf.config.list_physical_devices('GPU')
# if device:
#     try:
#         # 设置仅使用 GPU 1, 2, 3
#         tf.config.set_visible_devices(device[1:4], 'GPU')  # 索引从 0 开始
#         for gpu in device:
#             tf.config.experimental.set_memory_growth(gpu, True)  # 启用按需分配内存
#     except RuntimeError as e:
#         print(e)

def load_data(file_path, seq_len, pred_len):
    data = pd.read_csv(file_path)
    
    # 显式定义独热编码
    all_seasons = ['season_2', 'season_3', 'season_4']
    all_holidays = ['holiday_1']
    all_workingdays = ['workingday_1']
    all_weathersits = ['weathersit_2', 'weathersit_3', 'weathersit_4']

    data = pd.get_dummies(data, columns=['season', 'holiday', 'workingday', 'weathersit'], drop_first=True)

    # 缺失值补0
    for col in all_seasons + all_holidays + all_workingdays + all_weathersits:
        if col not in data.columns:
            data[col] = 0

    feature_columns = ['temp', 'atemp', 'hum', 'windspeed'] + all_seasons + all_holidays + all_workingdays + all_weathersits

    sequences = []
    casual_targets = []
    registered_targets = []

    # 滑动窗口
    for i in range(len(data) - seq_len - pred_len):
        seq_x = data.iloc[i:i + seq_len][feature_columns].values.astype(np.float32)
        casual_y = data.iloc[i + seq_len:i + seq_len + pred_len]['casual'].values.astype(np.float32)
        registered_y = data.iloc[i + seq_len:i + seq_len + pred_len]['registered'].values.astype(np.float32)
        sequences.append(seq_x)
        casual_targets.append(casual_y)
        registered_targets.append(registered_y)

    return np.array(sequences), np.array(casual_targets), np.array(registered_targets)

class BikeDataset(Dataset):
    def __init__(self, sequences, casual_targets, registered_targets):
        self.sequences = sequences
        self.casual_targets = casual_targets
        self.registered_targets = registered_targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.casual_targets[idx], dtype=torch.float32),
            torch.tensor(self.registered_targets[idx], dtype=torch.float32)
        )

class TransformerModel(nn.Module):
    def __init__(self, input_dim, pred_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward, dropout)
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.encoder(x) + self.positional_encoding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)
        x = self.decoder(x[:, -1, :])
        return x
    
# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_dim, pred_len, hidden_dim=64, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.decoder(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出进行预测
        return out
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size, padding=(kernel_size-1) * dilation_size),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class TCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels = [32, 64, 128]):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels)
        self.linear = nn.Linear(num_channels[-1], output_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # Convert to (batch_size, channels, sequence_length)
        x = self.tcn(x)
        x = x[:, :, -1]  # Take the last time step
        x = self.linear(x)
        return x

#使用L1计算一致性损失
def consistency_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = torch.abs(error) <= delta
    small_error_loss = 0.5 * error ** 2
    large_error_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.mean(torch.where(is_small_error, small_error_loss, large_error_loss))

def train_and_evaluate(trans_model, LSTM_model, TCN_model, train_loader, test_loader, trans_criterion, LSTM_criterion, TCN_criterion,trans_optimizer, LSTM_optimizer, TCN_optimizer,epochs, pred_len):
    trans_model.to(device)
    LSTM_model.to(device)
    TCN_model.to(device)
    history = {'train_loss': [], 'test_loss': [], 'mse': [], 'mae': [],
               'casual_preds': [], 'registered_preds': [], 'casual_targets': [], 'registered_targets': []}
    base_epoch = -1
    base_mae_epoch = 1000
    base_i = -1
    base_mae_i = 1000
    for epoch in range(epochs):
        trans_model.train()
        LSTM_model.train()
        train_loss = 0.0

        for x_batch, casual_y_batch, registered_y_batch in train_loader:
            x_batch = x_batch.to(device)
            casual_y_batch = casual_y_batch.to(device)
            registered_y_batch = registered_y_batch.to(device)

            trans_optimizer.zero_grad()
            LSTM_optimizer.zero_grad()

            # 获取 TCN 模型的预测结果
            TCN_registered_pred = TCN_model(x_batch)
            TCN_casual_pred = TCN_model(x_batch)

            registered_output = trans_model(x_batch)
            trans_loss = trans_criterion(registered_output, registered_y_batch)
        
            casual_output = LSTM_model(x_batch)
            LSTM_loss = LSTM_criterion(casual_output, casual_y_batch)

            registered_consistency = consistency_loss(TCN_registered_pred, registered_output)
            trans_loss += registered_consistency

            casual_consistency = consistency_loss(TCN_casual_pred, casual_output)
            LSTM_loss += casual_consistency

            trans_loss.backward()
            LSTM_loss.backward()
            trans_optimizer.step()
            LSTM_optimizer.step()

            train_loss += (trans_loss.item() + LSTM_loss.item()) * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Eval
        trans_model.eval()
        LSTM_model.eval()
        TCN_model.eval()
        test_loss = 0.0
        casual_preds, registered_preds, casual_targets, registered_targets = [], [], [], []

        with torch.no_grad():
            for i, (x_batch, casual_y_batch, registered_y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                casual_y_batch = casual_y_batch.to(device)
                registered_y_batch = registered_y_batch.to(device)

                # 获取 TCN 模型的预测结果
                TCN_registered_pred = TCN_model(x_batch)
                TCN_casual_pred = TCN_model(x_batch)

                registered_output = trans_model(x_batch)
                trans_loss = trans_criterion(registered_output, registered_y_batch)
            
                casual_output = LSTM_model(x_batch)
                LSTM_loss = LSTM_criterion(casual_output, casual_y_batch)

                registered_consistency = consistency_loss(TCN_registered_pred, registered_output)
                trans_loss += registered_consistency

                casual_consistency = consistency_loss(TCN_casual_pred, casual_output)
                LSTM_loss += casual_consistency

                test_loss += (trans_loss.item() + LSTM_loss.item()) * x_batch.size(0)
                
                for j in range(x_batch.size(0)):
                    mae = mean_absolute_error(casual_y_batch[j].cpu().numpy() + registered_y_batch[j].cpu().numpy(), casual_output[j].cpu().numpy() + registered_output[j].cpu().numpy())
                    if mae < base_mae_i:
                        base_mae_i = mae
                        base_i = i * x_batch.size(0) + j

                casual_preds.append(casual_output.cpu().numpy())
                registered_preds.append(registered_output.cpu().numpy())
                casual_targets.append(casual_y_batch.cpu().numpy())
                registered_targets.append(registered_y_batch.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        history['test_loss'].append(test_loss)

        casual_preds = np.concatenate(casual_preds)
        registered_preds = np.concatenate(registered_preds)
        casual_targets = np.concatenate(casual_targets)
        registered_targets = np.concatenate(registered_targets)

        history['casual_preds'].append(casual_preds)
        history['registered_preds'].append(registered_preds)
        history['casual_targets'].append(casual_targets)
        history['registered_targets'].append(registered_targets)

        mse = mean_squared_error(casual_targets + registered_targets, casual_preds + registered_preds)
        mae = mean_absolute_error(casual_targets + registered_targets, casual_preds + registered_preds)

        history['mse'].append(mse)
        history['mae'].append(mae)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

        if mae < base_mae_epoch:
            base_mae_epoch = mae.copy()
            base_epoch = epoch

    return history ,base_epoch, base_i,base_mae_i

def plot_prediction(original, predictions, ground_truth, title, save_path):
    plt.figure(figsize=(12, 8))

    # 绘制原始数据
    plt.plot(range(len(original)), original, label='Original Data', color='green', linestyle='--', alpha=0.8)

    # 绘制预测值和真实值
    prediction_start_idx = len(original)
    plt.plot(range(prediction_start_idx, prediction_start_idx + len(ground_truth)), ground_truth, label='Ground Truth', color='red', alpha=0.6)
    plt.plot(range(prediction_start_idx, prediction_start_idx + len(predictions)), predictions, label='Predictions', color='blue', alpha=0.6)

    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Bike Rentals')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main(train_path, test_path, seq_len, pred_len):
    # 加载数据
    train_sequences, train_casual_targets, train_registered_targets = load_data(train_path, seq_len, pred_len)
    test_sequences, test_casual_targets, test_registered_targets = load_data(test_path, seq_len, pred_len)

    train_dataset = BikeDataset(train_sequences, train_casual_targets, train_registered_targets)
    test_dataset = BikeDataset(test_sequences, test_casual_targets, test_registered_targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mse_results = []
    mae_results = []
    base_epoch = -1
    best_mae_i = 1000
    base_i = -1
    casual_original = 0
    casual_pred = 0
    casual_gt = 0
    registered_original = 0
    registered_pred = 0
    registered_gt = 0

    for i in range(5):  
        trans_model = TransformerModel(input_dim=train_sequences.shape[2], pred_len=pred_len)
        LSTM_model = LSTMModel(input_dim=train_sequences.shape[2], pred_len=pred_len)
        TCN_model = TCNModel(input_dim=train_sequences.shape[2], output_dim=pred_len)

        trans_criterion = nn.MSELoss()
        LSTM_criterion = nn.MSELoss()
        TCN_criterion = nn.MSELoss()

        trans_optimizer = torch.optim.Adam(trans_model.parameters(), lr=LEARNING_RATE)
        LSTM_optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=LEARNING_RATE)
        TCN_optimizer = torch.optim.Adam(TCN_model.parameters(), lr=LEARNING_RATE)
        
        history, epoch_i, i_i, mae_i = train_and_evaluate(trans_model, LSTM_model, TCN_model, train_loader, test_loader, trans_criterion, LSTM_criterion, TCN_criterion, trans_optimizer, LSTM_optimizer, TCN_optimizer,EPOCHS, pred_len)
        mse_results.append(history['mse'][-1])
        mae_results.append(history['mae'][-1])

        if mae_i < best_mae_i:
            best_mae_i = mae_i
            base_epoch = epoch_i
            base_i = i_i
            casual_original = history['casual_targets'][base_epoch][max(0,base_i - 96)]
            casual_pred = history['casual_preds'][base_epoch][base_i]
            casual_gt = history['casual_targets'][base_epoch][base_i]
            registered_original = history['registered_targets'][base_epoch][max(0,base_i - 96)]
            registered_pred = history['registered_preds'][base_epoch][base_i]
            registered_gt = history['registered_targets'][base_epoch][base_i]
        
    # 可视化预测结果
    plot_prediction(
        original=casual_original,
        predictions=casual_pred,
        ground_truth=casual_gt,
        title='Compare pred and GT',
        save_path='/data/zxzhou/gpt/ml/TCN/casual' + str(pred_len) + '.png'
    )

    plot_prediction(
        original=registered_original,
        predictions=registered_pred,
        ground_truth=registered_gt,
        title='Compare pred and GT',
        save_path='/data/zxzhou/gpt/ml/TCN/registered' + str(pred_len) + '.png'
    )

    plot_prediction(
        original=casual_original+registered_original,
        predictions=casual_pred + registered_pred,
        ground_truth=casual_gt + registered_gt,
        title='Compare pred and GT',
        save_path='/data/zxzhou/gpt/ml/TCN/cnt' + str(pred_len) + '.png'
    )

    mse_mean = np.mean(mse_results)
    mse_std = np.std(mse_results)
    mae_mean = np.mean(mae_results)
    mae_std = np.std(mae_results)

    print(f"Final Results - MSE: {mse_mean:.4f} (±{mse_std:.4f}), MAE: {mae_mean:.4f} (±{mae_std:.4f})")


main('/data/zxzhou/gpt/ml/train_data.csv', '/data/zxzhou/gpt/ml/test_data.csv', SEQ_LEN, SHORT_TERM_PRED)

main('/data/zxzhou/gpt/ml/train_data.csv', '/data/zxzhou/gpt/ml/test_data.csv', SEQ_LEN, LONG_TERM_PRED) 