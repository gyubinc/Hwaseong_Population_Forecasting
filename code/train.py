import torch
import torch.nn as nn
from .model import LSTM, TFModel
from .dataset import create_data_loader, Transformer_create_data_loader, windowDataset, Multi_WindowDataset
from .preprocessing import preprocessing
import pandas as pd
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from tqdm import tqdm


class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

class MAELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mae_loss = nn.functional.l1_loss(y_pred, y_true)
        return mae_loss


def LSTM_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')

    train_df, valid_df = preprocessing(df, args.window_size, args.step)
    
    
    train_loader = create_data_loader(train_df, args.window_size, args.batch_size, args.option, args.step)
    valid_loader = create_data_loader(valid_df, args.window_size, 1, args.option, args.step)
    

    
    # 모델 준비
    model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    model = model.float()
    
    # metric, optimizer 준비
    #criterion = nn.MSELoss()
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            
            labels = labels.to(device)
            # labels = labels.unsqueeze(1).to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)


            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
              
        wandb.log({"loss": loss.item(), "epoch": epoch,
            "learning_rate" : args.learning_rate
            })
        
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, args.num_epochs, loss.item()))

    print('*' * 30)
    print('valid test start')
    print(f'epochs : {args.num_epochs}')
    print(f'learning rate : {args.learning_rate}')
    print(f'batch size : {args.batch_size}')
    print(f'window size : {args.window_size}')
    print('*' * 30)
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(1).to(device)
            # Forward
            outputs = model(inputs)
            for i in range(args.step):
                val_pred = outputs[0][i]
                val_label = labels[0][i]
                loss = criterion(val_pred, val_label)
                L1loss =  MAE_criterion(val_pred, val_label)
                print(f'{i+1} 개월 RMSE loss : {round(loss.item(),4)}, MAE loss : {round(L1loss.item(), 4)}')
                wandb.log({"val_loss": loss.item()
                    })
                             
def Transformer_train(args):
    print('Time Series Transformer model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')

    train_df, valid_df = preprocessing(df, args.window_size, args.step)
    
    
    train_loader = Transformer_create_data_loader(train_df, args.window_size, args.batch_size, args.option, args.step)
    valid_loader = Transformer_create_data_loader(valid_df, args.window_size, 1, args.option, args.step)
    

    
    
    configuration = TimeSeriesTransformerConfig(prediction_length=1, input_size = args.step)
    model = TimeSeriesTransformerModel(configuration)
    configuration = model.config
    
    # metric, optimizer 준비
    #criterion = nn.MSELoss()
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        for i, (inputs, labels, time) in enumerate(train_loader):
            
            # breakpoint()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(1).to(device)
            breakpoint()
            # Forward
            outputs = model(
                past_values = inputs,
                past_time_features = time,
                past_observed_mask = np.zeros(inputs.shape),
                future_values = labels
                )
            loss = criterion(outputs, labels)


            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
              
        wandb.log({"loss": loss.item(), "epoch": epoch,
            "learning_rate" : args.learning_rate
            })
        
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, args.num_epochs, loss.item()))

    print('*' * 30)
    print('valid test start')
    print(f'epochs : {args.num_epochs}')
    print(f'learning rate : {args.learning_rate}')
    print(f'batch size : {args.batch_size}')
    print(f'window size : {args.window_size}')
    print('*' * 30)
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(1).to(device)
            # Forward
            outputs = model(inputs)
            for i in range(args.step):
                val_pred = outputs[0][i]
                val_label = labels[0][i]
                loss = criterion(val_pred, val_label)
                L1loss =  MAE_criterion(val_pred, val_label)
                print(f'{i+1} 개월 RMSE loss : {round(loss.item(),4)}, MAE loss : {round(L1loss.item(), 4)}')
                wandb.log({"val_loss": loss.item()
                    })
                
                
def uni_Transformer(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')

    data_train, data_test = preprocessing(df, args.window_size, args.step)
    data_train = data_train['총인구']
    data_test = data_test['총인구']
    
    iw = args.window_size
    ow = 12
    
    

    train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=4)
    
    valid_dataset = windowDataset(data_test, input_window=iw, output_window=ow, stride=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    
    lr = 1e-4
    model = TFModel(iw, ow, 256, 8, 4, 0.1).to(device)
    # breakpoint()
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 500
    model.train()
    progress = tqdm(range(epoch))
    
    
    for i in progress:
        batchloss = 0.0
        loss_back = []
        MAE_loss_back = []
        for (inputs, outputs) in train_loader:
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            loss = criterion(result, outputs[:,:,0].float().to(device))
            loss.backward()
            
            
            optimizer.step()
            batchloss += loss
            
            L1loss =  MAE_criterion(result, outputs[:,:,0].float().to(device))
            loss_back.append(loss.item())
            MAE_loss_back.append(L1loss.item())
            
            
        wandb.log({"mean_RMSE": np.mean(loss_back), 'mean_MAE': np.mean(MAE_loss_back),
                   "epoch": epoch, "learning_rate" : args.learning_rate,
                    })
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
    
    print('*' * 30)
    print('valid test start')
    print(f'epochs : {args.num_epochs}')
    print(f'learning rate : {args.learning_rate}')
    print(f'batch size : {args.batch_size}')
    print(f'window size : {args.window_size}')
    print('*' * 30)
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(1).to(device)
            # Forward
            
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            pred_list = []
            label_list = []
            for i in range(args.step):
                val_pred = result[0][i]
                val_label = labels[0][i]

                pred_list.append(val_pred)
                label_list.append(val_label)
                real_loss = val_pred - val_label
                print(f'{i+1} 개월 loss : {real_loss}')
                wandb.log({"val_loss": loss.item()
                    })
                
            loss = criterion(np.array(pred_list), np.array(label_list))
            L1loss =  MAE_criterion(np.array(pred_list), np.array(label_list))
            print(f'RMSE loss : {round(loss,1)}, MAE loss : {round(L1loss, 1)}')
            

def multi_Transformer(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')

    data_train, data_test = preprocessing(df, args.window_size, args.step)
    data_train = data_train
    data_test = data_test
    
    iw = args.window_size
    print(iw)
    ow = 12
    
    

    train_dataset = Multi_WindowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=4)
    
    valid_dataset = Multi_WindowDataset(data_test, input_window=iw, output_window=ow, stride=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    
    lr = 1e-4
    model = TFModel(iw*50, ow, 512, 8, 4, 0.1).to(device)
    # breakpoint()
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 500
    model.train()
    progress = tqdm(range(epoch))
    
    
    for i in progress:
        batchloss = 0.0
        loss_back = []
        MAE_loss_back = []
        for (inputs, outputs) in train_loader:
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            loss = criterion(result, outputs[:,:,0].float().to(device))
            loss.backward()
            
            
            optimizer.step()
            batchloss += loss
            
            L1loss =  MAE_criterion(result, outputs[:,:,0].float().to(device))
            loss_back.append(loss.item())
            MAE_loss_back.append(L1loss.item())
            
            
        wandb.log({"mean_RMSE": np.mean(loss_back), 'mean_MAE': np.mean(MAE_loss_back),
                   "epoch": epoch, "learning_rate" : args.learning_rate,
                    })
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
    
    print('*' * 30)
    print('valid test start')
    print(f'epochs : {args.num_epochs}')
    print(f'learning rate : {args.learning_rate}')
    print(f'batch size : {args.batch_size}')
    print(f'window size : {args.window_size}')
    print('*' * 30)
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(1).to(device)
            # Forward
            
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            pred_list = []
            label_list = []
            for i in range(args.step):
                val_pred = result[0][i]
                val_label = labels[0][i]

                pred_list.append(val_pred)
                label_list.append(val_label)
                real_loss = val_pred - val_label
                print(f'{i+1} 개월 loss : {real_loss}')
                wandb.log({"val_loss": loss.item()
                    })
                
            loss = criterion(np.array(pred_list), np.array(label_list))
            L1loss =  MAE_criterion(np.array(pred_list), np.array(label_list))
            print(f'RMSE loss : {round(loss,1)}, MAE loss : {round(L1loss, 1)}')