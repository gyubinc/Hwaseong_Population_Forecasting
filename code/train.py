import wandb
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error 

from .model import LSTM, TFModel
from .dataset import create_data_loader, windowDataset, Multi_WindowDataset
from .preprocessing import preprocessing

# RMSE class 정의
class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

# MAELoss class 정의
class MAELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mae_loss = nn.functional.l1_loss(y_pred, y_true)
        return mae_loss

# 단변량 LSTM model train
def LSTM_train(args):
    # CPU / GPU 탑재
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')
    train_df, valid_df = preprocessing(df, args.window_size, args.step)
    
    # train / valid set 정의
    train_loader = create_data_loader(train_df, args.window_size, args.batch_size, args.option, args.step)
    valid_loader = create_data_loader(valid_df, args.window_size, 1, args.option, args.step)

    # 모델 준비
    model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    model = model.float()

    # 모델 옵션 및 하이퍼 파라미터 정의
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 모델 학습
    for epoch in range(args.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)      
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
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
    
    # valid
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred_list = []
            label_list = []
            for i in range(args.step):
                val_pred = outputs[0][i]
                val_label = labels[0][i]
                pred_list.append(val_pred)
                label_list.append(val_label)
                loss = criterion(val_pred, val_label)
                L1loss =  MAE_criterion(val_pred, val_label)
                print(f'{val_pred - val_label}')
            loss = np.sqrt(mean_squared_error(pred_list, label_list))
            L1loss =  mean_absolute_error(pred_list, label_list)
            print(f'RMSE loss : {round(loss,1)}, MAE loss : {round(L1loss, 1)}')                

# U_TFT
def uni_Transformer(args):
    # GPU / CPU 탑재
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 모델 하이퍼 파라미터 정의
    iw = args.window_size
    ow = args.output_size
    batch_size = args.batch_size
    lr = args.learning_rate
    epoch = args.num_epochs
    step = args.step
    
    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')
    data_train, data_test = preprocessing(df, iw, step)
    data_train = data_train['총인구']
    data_test = data_test['총인구']
    
    # train/ valid set 정의
    train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset = windowDataset(data_test, input_window=iw, output_window=ow, stride=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    
    # 모델 정의
    model = TFModel(iw, ow, 256, 8, 4, 0.1).to(device)
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    print('*' * 30)
    print('train start')
    print(f'epochs : {epoch}')
    print(f'learning rate : {lr}')
    print(f'batch size : {batch_size}')
    print(f'window size : {iw}')
    print('*' * 30)   
    
    # 모델 학습
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
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
    

    # 모델 validation
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            pred_list = []
            label_list = []
            for i in range(args.step):
                val_pred = result[0][i]
                val_label = labels[0][i]
                pred_list.append(int(val_pred))
                label_list.append(int(val_label))
                real_loss = val_pred - val_label
                print(f'{i+1} 개월 loss : {real_loss}')
            loss = np.sqrt(mean_squared_error(pred_list, label_list))
            L1loss =  mean_absolute_error(pred_list, label_list)
            print(f'RMSE loss : {round(loss,1)}, MAE loss : {round(L1loss, 1)}')
            
def multi_Transformer(args):
    # argumnets 정의
    lr = args.learning_rate
    epoch = args.num_epochs
    iw = args.window_size
    batch_size = args.batch_size
    ow = args.output_size
    
    # GPU / CPU 탑재
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')
    data_train, data_test = preprocessing(df, args.window_size, args.step)
    data_train = data_train
    data_test = data_test

    # train / valid set 정의
    train_dataset = Multi_WindowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset = Multi_WindowDataset(data_test, input_window=iw, output_window=ow, stride=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # 모델 및 하이퍼 파라미터 정의
    model = TFModel(iw*50, ow, 128, 8, 4, 0.1).to(device)
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print('*' * 30)
    print('train start')
    print(f'epochs : {epoch}')
    print(f'learning rate : {lr}')
    print(f'batch size : {batch_size}')
    print(f'window size : {iw}')
    print('*' * 30)
    
    # 모델 학습
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
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
    
    # 모델 평가
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            pred_list = []
            label_list = []
            print(result)
            for i in range(args.step):
                val_pred = result[0][i]
                val_label = labels[0][i][0]
                print(val_pred)
                print(val_label)
                pred_list.append(int(val_pred))
                label_list.append(int(val_label))
                real_loss = val_pred - val_label
                print(f'{i+1} 개월 loss : {real_loss}')
            loss = np.sqrt(abs(mean_squared_error(pred_list, label_list)))
            L1loss =  mean_absolute_error(pred_list, label_list)
            print(f'RMSE loss : {round(loss,1)}, MAE loss : {round(L1loss, 1)}')
            
def CPU_multi_Transformer(args):
    # arguments 정의
    lr = args.learning_rate
    epoch = args.num_epochs
    iw = args.window_size
    batch_size = args.batch_size
    ow = args.output_size
    
    # CPU / GPU 탑재
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')
    data_train, data_test = preprocessing(df, args.window_size, args.step)

    # train / valid set 정의
    train_dataset = Multi_WindowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset = Multi_WindowDataset(data_test, input_window=iw, output_window=ow, stride=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # 모델 및 하이퍼 파라미터 정의    
    model = TFModel(iw*50, ow, 128, 8, 4, 0.1).to(device)
    criterion = RMSELoss()
    MAE_criterion = MAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.995 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    
    print('*' * 30)
    print('train start')
    print(f'epochs : {epoch}')
    print(f'learning rate : {lr}')
    print(f'batch size : {batch_size}')
    print(f'window size : {iw}')
    print('*' * 30)
    
    # 모델 학습
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
        scheduler.step()    
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
    
    
    # 모델 validation
    device = 'cpu'
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device),  src_mask)
            pred_list = []
            label_list = []
            loss_list = []
            print(result)
            for j in range(args.step):
                val_pred = result[0][j]
                val_label = labels[0][j][0]
                real_pred = float(val_pred)
                real_label = float(val_label)
                print(f'{j+1} 개월 예측 : {int(real_pred)}명')
                print(f'{j+1} 개월 정답 : {int(real_label)}명')
                pred_list.append(real_pred)
                label_list.append(real_label)
                real_loss = real_pred - real_label
                loss_list.append(real_loss)
                print(f'{j+1} 개월 loss : {real_loss}')
            loss = np.sqrt(abs(mean_squared_error(pred_list, label_list)))
            L1loss =  mean_absolute_error(pred_list, label_list)
            print(f'RMSE loss : {round(loss,1)}, MAE loss : {round(L1loss, 1)}')