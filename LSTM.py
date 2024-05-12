import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SteamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def prepare_data(csv_file, N):
    data = pd.read_csv(csv_file)
    data.sort_values('date', inplace=True)
    feature_columns = [col for col in data.columns if col not in ['avg', 'date']]
    target_column = 'avg'

    def create_sequences(data, N):
        X, y = [], []
        features = data[feature_columns].values
        labels = data[target_column].values
        for i in range(len(features) - N):
            X.append(features[i:i+N])
            y.append(labels[i+N])
        return np.array(X), np.array(y)

    X, y = create_sequences(data, N)
    return X, y

N = 12
X, y = prepare_data('/kaggle/input/steam-data-1/clean_data.csv', N)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_prob, 
                            bidirectional=True).to(device)
        self.fc = nn.Linear(hidden_size * 2, output_size).to(device)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out_forward = out[:, -1, :self.hidden_size]
        out_reverse = out[:, 0, self.hidden_size:]
        out_concatenated = torch.cat((out_forward, out_reverse), dim=1)
        out = self.fc(out_concatenated)
        return out


tscv = TimeSeriesSplit(n_splits=5)
total_rmse = 0

patience = 200
epochs_no_improve = 0
epochs = 9000

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = SteamDataset(X_train, y_train)
    test_dataset = SteamDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    model = LSTMModel(input_size=len(X_train[0][0]), hidden_size=20, num_layers=10, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                total_eval_loss += loss.item()
        
        average_eval_loss = total_eval_loss / len(test_loader)

        if epoch % 100 == 0:
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {average_train_loss:.4f}, Eval Loss: {average_eval_loss:.4f}')

        # Check for early stopping
        if average_eval_loss < best_loss:
            best_loss = average_eval_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth') 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    
    
    model.load_state_dict(torch.load('best_model.pth'))
    rmse = np.sqrt(best_loss)
    total_rmse += rmse
    print(f'Fold {fold + 1}, Test RMSE: {rmse}')

average_rmse = total_rmse / 5
print(f'Average Test RMSE: {average_rmse}')
