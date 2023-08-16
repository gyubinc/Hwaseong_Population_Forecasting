import torch
import torch.nn as nn
from .model import LSTM
from .dataset import create_data_loader
from .preprocessing import preprocessing
import pandas as pd

class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse_loss = nn.functional.mse_loss(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss


def LSTM_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # 데이터 준비
    df = pd.read_excel(args.train_path)
    df = preprocessing(df)
    train_loader = create_data_loader(df, args.window_size, args.batch_size)
    
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
            
            if (i == 1 and  epoch % 100 == 1):
                print('*'*30)
                print(f'{loss}, {outputs}, {labels}')
                print('*'*30)
            
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, args.num_epochs, loss.item()))
