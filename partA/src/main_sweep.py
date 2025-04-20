from train_sweep import *
# import libraries
import wandb

# best_acc_sweep_configuration = {
#     "method": "random",
#     "name" : "test_sweep1",
#     "parameters": {
#         "num_filters": {'values': [[256, 128, 64, 64, 32]]},
#         "filter_sizes": {'values': [[3, 3, 3, 3, 3]]},
#         "activation": {"values": ["SiLU"]},
#         "optimizer": {"values": ["sgd"]},
#         "learning_rate": {"values": [1e-3]},
#         "weight_decay": {"values": [0.0001]},
#         "momentum": {"values": [0.9]},
#         "beta": {"values": [0.9]},
#         "beta1": {"values":[0.9]},
#         "beta2": {"values": [0.999]},
#         "epsilon": {"values": [1e-8]},
#         # "base_dir": {"values":["/content/drive/MyDrive/DL_Assignment2/Dataset/inaturalist_12K/"]},
#         "base_dir": {"values": ["/kaggle/input/inaturalist/inaturalist_12K"]},
#         "isDataAug": {"values": ["False"]},
#         "isBatchNormalization": {"values": ["False"]},
#         "dropout": {"values": [0.3]},
#         "n_neurons_denseLayer": {"values": [128]},
#         "batch_size": {"values": [32]},
#         "epochs": {"values": [10]}
#     }
# }

sweep_configuration = {
    "method": "random",
    "name" : "train_sweep_final2_v1",
    "parameters": {
        "num_filters": {'values': [[32, 32, 32, 32, 32], [32, 64, 64, 128, 256], [256, 128, 64, 64, 32]]},
        "filter_sizes": {'values': [[3, 3, 3, 3, 3], [5, 5, 5, 5, 5],[3,3,5,5,1]]},
        "activation": {"values": ["ReLU", "SiLU", "GELU"]},
        "optimizer": {"values": ["adam", "rmsprop", "sgd"]},
        "learning_rate": {"values": [1e-3]},
        "weight_decay": {"values": [0.0001]},
        "momentum": {"values": [0.9]},
        "beta": {"values": [0.9]},
        "beta1": {"values":[0.9]},
        "beta2": {"values": [0.999]},
        "epsilon": {"values": [1e-8]},
        "base_dir": {"values":["/content/drive/MyDrive/DL_Assignment2/Dataset/inaturalist_12K/"]},
        # "base_dir": {"values": ["/kaggle/input/inaturalist/inaturalist_12K"]},
        "isDataAug": {"values": ["False", "True"]},
        "isBatchNormalization": {"values": ["True", "False"]},
        "dropout": {"values": [0.2, 0.3]},
        "n_neurons_denseLayer": {"values": [128, 256]},
        "batch_size": {"values": [32,64]},
        "epochs": {"values": [5,10]}
    }
}

if __name__=="__main__":
  wandb.login()
  wandb_id = wandb.sweep(sweep_configuration, project="DA6401_Assignment2")
  wandb.agent(wandb_id, function=trainNeuralNetwork_sweep)
