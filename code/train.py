import torch
import torch.nn as nn
from .model import LSTM
from .dataset import create_data_loader
from .preprocessing import preprocessing
import pandas as pd
import wandb

class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss


def LSTM_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path, index_col = '월별')
    
    train_df, valid_df = preprocessing(df, args.window_size)
    
    
    train_loader = create_data_loader(train_df, args.window_size, args.batch_size, args.option, args.step)
    valid_loader = create_data_loader(valid_df, args.window_size, 1, args.option, args.step)
    

    
    # 모델 준비
    model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size).to(device)
    model = model.float()
    
    # metric, optimizer 준비
    #criterion = nn.MSELoss()
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # if (i == 1 and  epoch % 100 == 1):
            #     print('*'*30)
            #     print(f'loss : {loss}\n\
            #           outputs : {outputs}\n\
            #           labels : {labels}')
            #     print('*'*30)
            
            
            # Backward and optimize


            
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
            labels = labels.unsqueeze(1).to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(f'{i+1} 개월 loss : {round(loss.item(),4)}')
            wandb.log({"val_loss": loss.item()
                })
        