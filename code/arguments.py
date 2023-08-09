import argparse

def get_args():
    # 기본 설정값만 남기고, hyperparameter는 wandb에게 맡김.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='LSTM', type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--loss', default='L1', help="['L1','MSE','Huber','plcc'] 중 택1", type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')

    parser.add_argument('--input_size', defalut = 8, type = int)
    parser.add_argument('--hidden_size', default = 64, type = int)
    parser.add_argument('--num_layers', defalut = 2, type = int)
    parser.add_argument('--output_size', defalut = 1, type = int)
    parser.add_argument('--window_size', defalut = 24, type = int)
    parser.add_argument('--batch_size', defalut = 64, type = int)
    
    args = parser.parse_args(args=[])
    return args