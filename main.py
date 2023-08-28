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
    "epochs": 2000,
    "learning_rate": 1e-2,
    "batch_size": 4,
    "window_size": 20
    }
    
    args_dict = vars(args)
    
    wandb.config.update(args_dict)
    
    LSTM_train(args)
    print('finished')
    
    