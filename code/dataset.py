import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

'''
종속변수는 항상 마지막 열에 위치시키고 시작
uni-variate 데이터는 무조건 one-step 기준

option
1 = univariate
2 = multivariate

step
1 = one-step
n = multi-step (n-step)

'''


class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size, option, step):
        self.df = df
        self.window_size = window_size
        self.option = option
        self.step = step

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # multivariate
        if self.option == 2:
            # one-step
            if self.step == 1:
                x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)
                if self.df.shape[1] > 1:
                    y = torch.tensor(self.df.iloc[idx+self.window_size, -1], dtype=torch.float32)
                else:
                    y = None
                return x, y
            # multi-step
            else:
                x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)
                if self.df.shape[1] > 1:
                    y = torch.tensor(self.df.iloc[idx+self.window_size : idx+self.window_size + self.step, -1], dtype=torch.float32)
                else:
                    y = None
                return x, y
        
        # univariate, one-step
        elif self.option == 1:
            x = torch.tensor(self.df.iloc[idx:idx+self.window_size, -1].to_numpy(), dtype=torch.float32)
            if self.df.shape[1] > 1:
                y = torch.tensor(self.df.iloc[idx+self.window_size, -1], dtype=torch.float32)
            else:
                y = None
            return x, y
            

def create_data_loader(df, window_size, batch_size, option, step):
    dataset = TimeSeriesDataset(df, window_size, option, step)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader