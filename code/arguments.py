import argparse

def get_args():
    # parser 정의
    parser = argparse.ArgumentParser()
    
    # 모델 정의
    parser.add_argument('--model_name', default='LSTM', type=str)
    parser.add_argument('--input_size', default = 50, type = int)
    parser.add_argument('--hidden_size', default = 64, type = int)
    parser.add_argument('--num_layers', default = 4, type = int)
    parser.add_argument('--output_size', default = 12, type = int)
    parser.add_argument('--window_size', default = 24, type = int)
    
    # 모델 하이퍼 파라미터 정의
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=2000, type=int)
    parser.add_argument('--shuffle', default=True)
    
    # 학습 옵션 정의
    parser.add_argument('--train_path', default='./data/data.xlsx')
    parser.add_argument('--option', default = 2, type = int)
    parser.add_argument('--step', default = 12, type = int)
    parser.add_argument('--inference', default = 0, type = int)
    
    return parser.parse_args()