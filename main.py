from code.utils import seed_everything
from code.train import LSTM_train,  uni_Transformer, multi_Transformer, CPU_multi_Transformer
from code.arguments import get_args
from code.inference import inference


if __name__ == "__main__":
    # arguments 정의
    args = get_args()
    
    # seed 고정
    seed_everything(42)

    # LSTM_train(args)
    # uni_Transformer(args)
    if args.inference != 0:
        inference(args)
    else :
        CPU_multi_Transformer(args)
    
    print('train finished')
    
    