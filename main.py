from code.utils import seed_everything
from code.train import LSTM_train,  uni_Transformer, multi_Transformer, CPU_multi_Transformer
from code.arguments import get_args
import wandb

if __name__ == "__main__":
    # seed 고정
    args = get_args()
    seed_everything(42)
    
    
    # # WandB
    # wandb.login(key = '28c2410815e7aa7e1b762a66d5dc91dc8edb48d8' )
    # wandb.init(project='Hwaseong_Population_Forecasting')
    
    # wandb.run.name = 'TimeSeries_Transformer_01'
    # wandb.run.save()
    
    # wandb.config = {
    # "num_epochs": args.num_epochs,
    # "learning_rate": args.learning_rate,
    # "batch_size": args.batch_size,
    # "window_size": args.window_size
    # }
    
    # LSTM_train(args)
    # multi_Transformer(args)
    CPU_multi_Transformer(args)
    # uni_Transformer(args)
    
    print('finished')
    
    