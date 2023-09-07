import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from torch.utils.data import DataLoader, Dataset

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
        return len(self.df) - self.window_size - self.step + 1

    def __getitem__(self, idx):
        # multivariate
        if self.option == 2:
            # one-step
            if self.step == 1:

                x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)
                if self.df.shape[1] > 1:
                    y = torch.tensor(self.df.iloc[idx+self.window_size, 0], dtype=torch.float32)
                else:
                    y = None
                return x, y
            # multi-step
            else:
                x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)

                if self.df.shape[1] > 1:

                    y = torch.tensor(self.df.iloc[idx+self.window_size : idx+self.window_size + self.step, 0].to_numpy(), dtype=torch.float32)
                    
                else:
                    print('x')
                    y = None
                return x, y
        
        # univariate, one-step
        elif self.option == 1:
            x = torch.tensor(self.df.iloc[idx:idx+self.window_size, 0].to_numpy(), dtype=torch.float32)
            if self.df.shape[1] > 1:
                y = torch.tensor(self.df.iloc[idx+self.window_size, 0], dtype=torch.float32)
            else:
                y = None
            return x, y
            

def create_data_loader(df, window_size, batch_size, option, step):
    dataset = TimeSeriesDataset(df, window_size, option, step)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader




class Transformer_TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size, option, step):
        self.df = df
        self.window_size = window_size
        self.option = option
        self.step = step

    def __len__(self):
        return len(self.df) - self.window_size - self.step + 1

    def __getitem__(self, idx):
        # multivariate
        if self.option == 2:
            # one-step
            if self.step == 1:

                x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)
                if self.df.shape[1] > 1:
                    y = torch.tensor(self.df.iloc[idx+self.window_size, 0], dtype=torch.float32)
                else:
                    y = None
                return x, y
            # multi-step
            else:
                x = torch.tensor(self.df.iloc[idx:idx+self.window_size, :].to_numpy(), dtype=torch.float32)
                time = []
                for i in range(self.window_size):
                    tm = []
                    for j in range(len(x[0])):
                        t = int(str(self.df.index[idx+i])[0:4] + str(self.df.index[idx+i])[5:7] + str(j) )
                        tm.append(t)
                    time.append(tm)
                time = torch.tensor(time)
                # breakpoint()
                if self.df.shape[1] > 1:

                    y = torch.tensor(self.df.iloc[idx+self.window_size : idx+self.window_size + self.step, 0].to_numpy(), dtype=torch.float32)
                    
                else:
                    print('x')
                    y = None
                return x, y, time
        
        # univariate, one-step
        elif self.option == 1:
            x = torch.tensor(self.df.iloc[idx:idx+self.window_size, 0].to_numpy(), dtype=torch.float32)
            if self.df.shape[1] > 1:
                y = torch.tensor(self.df.iloc[idx+self.window_size, 0], dtype=torch.float32)
            else:
                y = None
            return x, y
        
        
        
def Transformer_create_data_loader(df, window_size, batch_size, option, step):
    dataset = Transformer_TimeSeriesDataset(df, window_size, option, step)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader        


'''
단변량 transformer 구현
'''
class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

class Multi_WindowDataset(Dataset):
    def __init__(self, x, input_window=80, output_window=20, stride=5):
        #총 데이터의 개수
        y = x['총인구']
        # columns_num = len(x.iloc[0,:])
        columns_num = 50

        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window * columns_num, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            data_list = []
            for j in range(columns_num):
                data_list.append(x.iloc[start_x:end_x, j])
            X[:,i] = np.concatenate(data_list)

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]
        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y
        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len