import os
import gc
import wandb
import torch
from neural_network import *
from data_loader import *
from accuracy_calculation import *
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def trainNeuralNetwork_local(args):
  wandb.login()
  wandb.init(mode="online")
  wandb.init(project=args.wandb_project, entity=args.wandb_entity)
  if args.isDataAug == "True":
    isDataAug = True
  else:
    isDataAug = False
  
  if args.isBatchNormalization == "True":
    isBatchNormalization = True
  else:
    isBatchNormalization = False

  train_loader, test_loader, val_loader, num_classes = load_data(args.base_dir, isDataAug, args.batch_size)
  activationFun = args.activation
  optimizer = args.optimizer
  learning_rate = args.learning_rate
  momentum = args.momentum
  beta = args.beta
  beta1 = args.beta1
  beta2 = args.beta2
  epsilon = args.epsilon
  weight_decay = args.weight_decay
  dropout = args.dropout
  num_filters = args.num_filters
  filter_sizes = args.filter_sizes
  n_neurons_denseLayer = args.n_neurons_denseLayer

  wandb.run.name = f"{activationFun}_{optimizer}_{dropout}_{n_neurons_denseLayer}_DataAug-{isDataAug}_BatchNorm-{isBatchNormalization}"
  best_val_accuracy = 0.0
  best_accuracy_epoch = -1

  cnn = ConvolutionalNeuralNetwork(num_classes,
                                   num_filters, filter_sizes,
                                   activationFun, optimizer,
                                   n_neurons_denseLayer,
                                   isBatchNormalization, dropout,
                                   learning_rate,
                                   momentum, beta,
                                   beta1, beta2,
                                   epsilon, weight_decay)
  cnn.to(device)

  epochs = args.epochs
  for epochNum in range(epochs):
    print(f"Epoch {epochNum}:")
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
      if(batch_idx % 40 == 0):
        print(f"Batch idx {batch_idx} running")
        # break
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      cnn.optimizer.zero_grad()
      outputs = cnn(x_batch)
      cnn.backward(outputs, y_batch)
      cnn.updateWeights()
      del x_batch, y_batch, outputs

    # Validation accuracy
    val_outputs, val_accuracy, val_loss, _, _ = findOutputs(device, cnn, val_loader)
    print(f"validation: loss={val_loss}, accuracy={val_accuracy}")

    # Train accuracy
    train_outputs, train_accuracy, train_loss, _, _ = findOutputs(device, cnn, train_loader)
    print(f"training: loss={train_loss}, accuracy={train_accuracy}")

    if val_accuracy > best_val_accuracy:
      best_val_accuracy = val_accuracy
      best_accuracy_epoch = epochNum

    wandb.log({
        "epoch": epochNum + 1,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy
        },commit=True)
    del val_outputs, train_outputs
    gc.collect()
    torch.cuda.empty_cache()      
      
  wandb.log({
      "best_acc_epoch": best_accuracy_epoch,
      "best_val_accuracy": best_val_accuracy
  })

  test_outputs, test_accuracy, test_loss, test_top5_accuracy, test_top2_accuracy = findOutputs(device, cnn, test_loader, True)
  print(f"testing: loss={test_loss}, top1_accuracy={test_accuracy}, top5_accuracy = {test_top5_accuracy}, top2_accuracy = {test_top2_accuracy}")

  wandb.log({
      "test_loss": test_loss,
      "test_top1_accuracy": test_accuracy,
      "test_top5_accuracy": test_top5_accuracy,
      "test_top2_accuracy": test_top2_accuracy
  })
  del cnn,train_loader, test_loader, val_loader
  gc.collect()
  torch.cuda.empty_cache()

  wandb.finish()
