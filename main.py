from code.utils import seed_everything
from code.train import LSTM_train
from code.arguments import get_args
import wandb

if __name__ == "__main__":
    # seed 고정
    seed_everything(42)
    args = get_args()
    
    # WandB
    wandb.login(key = 'your api key' )
    wandb.init(project='Hwaseong_Population_Forecasting')
    
    wandb.run.name = 'Experiment_01'
    wandb.run.save()
    
    wandb.config = {
    "num_epochs": args.num_epochs,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "window_size": args.window_size
    }
    
    LSTM_train(args)
    print('finished')
    
    