from code.utils import seed_everything
from code.train import LSTM_train
from code.arguments import get_args

if __name__ == "__main__":
    # seed 고정
    seed_everything(42)
    args = get_args()
    
    LSTM_train(args)
    print('finished')
    
    