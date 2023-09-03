import argparse

def get_args():
    # 기본 설정값만 남기고, hyperparameter는 wandb에게 맡김.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='LSTM', type=str)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--loss', default='L1', help="['L1','MSE','Huber','plcc'] 중 택1", type=str)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=2000, type=int)
    
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='./data/찐 최종 데이터.xlsx')
    
    parser.add_argument('--dev_path', default='./data/semi_last.xlsx')
    parser.add_argument('--test_path', default='./data/semi_last.xlsx')
    parser.add_argument('--predict_path', default='./data/semi_last.xlsx')

    parser.add_argument('--input_size', default = 50, type = int)
    parser.add_argument('--hidden_size', default = 64, type = int)
    parser.add_argument('--num_layers', default = 4, type = int)
    parser.add_argument('--output_size', default = 12, type = int)
    parser.add_argument('--window_size', default = 24, type = int)
    
    parser.add_argument('--option', default = 2, type = int)
    parser.add_argument('--step', default = 12, type = int)
    
    
    args = parser.parse_args(args=[])
    return args