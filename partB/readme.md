# Convolutional Neural Network - Finetuning the pretrained model

This repository implements a Convolutional Neural Network (CNN) to classify images from iNaturalist dataset using a pretrained model by finetuning it. </br>
Pre-trained model considered is "EfficientNet_V2_S"([link](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.EfficientNet_V2_S_Weights)) </br>
Implementation can be found in python notebook and also in different files across directory as mentioned below.

## Code Organization

```
├── dl-assignment2-partb-wandb.ipynb       # Notebook to train via W&B without requirement of src files
├── dl_assignment2_partb_local.ipynb       # Organization of src files and its running can also be understood from this notebook
└── src/
    ├── accuracy_calculation.py            # Evalutes model and computes accuracy (top1,top2 etc..)
    ├── argument_parser.py                 # Parses command-line arguments (only for single run using main.py)
    ├── data_loader.py                     # Loads and preprocesses dataset (as per the pretrained transforms)
    ├── libraries.py                       # Contains all library imports.
    ├── main.py                            # Entry point(main script) for single run
    ├── main_sweep.py                      # Entry point for sweep runs  (sweep should be modified in main_sweep.py only)
    ├── neural_network.py                  # Defines CNN model from the pretrained model (fine tunes here)
    ├── train_local.py                     # Training of single run (as per main.py)
    └── train_sweep.py                     # Training with hyperparameter sweeps (as per main_sweep.py)
```

## Finetune Strategy Implemented
Freezing all layers except the last fully connected classifier layer.

## Running the Training Script
As clear from the directory structure that, one can run locally from src files as a single run and sweep using main.py and main_sweep.py

-- To train the CNN for a single run using commond line arguments:

```bash
python main.py --wandb_entity nikhithaa-iit-madras --wandb_project DA6401_Assignment2
```
Along with the wandb entity, project different arguments are supported from the command line as shown below. </br>
Some of the partA parameters aren't part of these since we are using pre-trained model and fine tuning it.


### Supported Arguments for main.py

| Shortcut Argument | Argument | Default | Description |
| ----------------- |---------|---------|-------------|
| `-wb` | `--wandb_project` | DA6401_Assignment2 | WandB project name |
| `-we` | `--wandb_entity` | nikhithaa-iit-madras | WandB entity |
| `-bd` | `--base_dir` | inaturalist_12K | Dataset directory (with `train/` and `val/`) |
| `-e` | `--epochs` | 10 | Number of training epochs |
| `-b` | `--batch_size` | 32 | Batch size |
| `-o` | `--optimizer` | sgd | [`sgd`, `rmsprop`, `adam`] |
| `-lr` | `--learning_rate` | 0.001 | Learning rate |
| `-m` | `--momentum` | 0.9 | Momentum (for `sgd`) |
| `-beta` | `--beta` | 0.9 | Beta (for `rmsprop`) |
| `-beta1` | `--beta1` | 0.9 | Beta1 (for `adam`) |
| `-beta2` | `--beta2` | 0.999 | Beta2 (for `adam`) |
| `-eps` | `--epsilon` | 1e-8 | Small constant for numerical stability |
| `-w_d` | `--weight_decay` | 0.0001 | L2 regularization |
| `-da` | `--isDataAug` | False | Whether to use data augmentation along with pretrained transforms|

-- To train the CNN for a sweep: (Sweep configuration should be modified in the main_sweep.py)

```bash
python main_sweep.py 
```

## 🔗 Useful Links
- **Wandb Report**: https://wandb.ai/nikhithaa-iit-madras/DA6401_Assignment2/reports/DA6401-Assignment-2--VmlldzoxMjI0NTEwMA