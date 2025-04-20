from train_sweep import *
# import libraries
import wandb

sweep_configuration = {
    "method": "random",
    "name" : "finetune_final_sweep",
    "parameters": {
        "num_filters": {'values': [[32, 32, 32, 32, 32]]},  # have no effect since using pre-trained model
        "filter_sizes": {'values': [[3, 3, 3, 3, 3]]},        # have no effect since using pre-trained model
        "activation": {"values": ["ReLU"]},   # have no effect since using pre-trained model
        "optimizer": {"values": ["adam", "rmsprop", "sgd"]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "weight_decay": {"values": [0.0001, 0.0005]},
        "momentum": {"values": [0.9]},
        "beta": {"values": [0.9]},
        "beta1": {"values":[0.9]},
        "beta2": {"values": [0.999]},
        "epsilon": {"values": [1e-8]},
        # "base_dir": {"values":["/content/drive/MyDrive/DL_Assignment2/Dataset/inaturalist_12K/"]},
        "base_dir": {"values": ["../Dataset/inaturalist_12K"]},          
        "isDataAug": {"values": ["False", "True"]},
        "isBatchNormalization": {"values": ["False"]},  # have no effect since using pre-trained model
        "dropout": {"values": [0]},            # have no effect since using pre-trained model
        "n_neurons_denseLayer": {"values": [128]},   # have no effect since using pre-trained model
        "batch_size": {"values": [32,64]},
        "epochs": {"values": [5,10]}
    }
}

if __name__=="__main__":
  wandb.login()
  wandb_id = wandb.sweep(sweep_configuration, project="DA6401_Assignment2")
  wandb.agent(wandb_id, function=trainNeuralNetwork_sweep)
