import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size):
        self.df = df
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        
        x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)
        if self.df.shape[1] > 1:
            y = torch.tensor(self.df.iloc[idx+self.window_size, -1], dtype=torch.float32)
        else:
            y = None
        return x, y

def create_data_loader(df, window_size, batch_size):
    dataset = TimeSeriesDataset(df, window_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader