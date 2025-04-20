import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class ConvolutionalNeuralNetwork(nn.Module):
  activationFunctionsMap = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}
  # optimizersMap = {"sgd": optim.SGD, "rmsprop": optim.RMSprop, "adam": optim.Adam}

  def __init__(self, num_classes,
               num_filters, filter_sizes,
               activationFun, optimizer,
               n_neurons_denseLayer,
               isBatchNormalization, dropout,
               learning_rate=0.001,
               momentum=0.5, beta = 0.9,
               beta1=0.9, beta2=0.99,
               epsilon=1e-8, weight_decay=0.0001):
    super(ConvolutionalNeuralNetwork, self).__init__()
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.filter_sizes = filter_sizes
    self.activationFun = ConvolutionalNeuralNetwork.activationFunctionsMap[activationFun]
    # self.optimizer = ConvolutionalNeuralNetwork.optimizersMap[optimizer]

    self.n_neurons_denseLayer = n_neurons_denseLayer
    self.isBatchNormalization = isBatchNormalization
    self.dropout = dropout

    self.lr = learning_rate
    self.momentum = momentum
    self.betas = (beta1, beta2)
    self.eps = epsilon
    self.alpha = beta
    self.weight_decay = weight_decay
    # self.count = 0
    # self.cached_inputs = None    # dataloader shuffling within batches to cache (else needs to make shuffle = False)

    self.defineModel()

    trainable_parameters = [p for p in self.parameters() if p.requires_grad == True]

    if(optimizer == "sgd"):
      self.optimizer = optim.SGD(trainable_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
    elif(optimizer == "rmsprop"):
      self.optimizer = optim.RMSprop(trainable_parameters, lr=self.lr, alpha=self.alpha, eps=self.eps, weight_decay=self.weight_decay)
    elif(optimizer == "adam"):
      self.optimizer = optim.Adam(trainable_parameters, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
    # print(trainable_parameters)

  def defineModel(self):
    self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")

    # Freezing all layers except the last layer
    for param in self.model.parameters():
        param.requires_grad = False

    # Replacing last layer (its a classifier at the end containing dropout followed by linear layer)
    num_in_features_last_layer = self.model.classifier[1].in_features
    self.model.classifier[1] = nn.Linear(num_in_features_last_layer, self.num_classes)

    # last layer is trainable
    for name, param in self.model.classifier[1].named_parameters():
        param.requires_grad = True 
    
  def forward(self, inputs):
    return self.model(inputs)
      
  def backward(self, outputs, labels):
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

  def updateWeights(self):
    self.optimizer.step()
