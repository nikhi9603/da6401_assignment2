import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

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

    self.defineModel()

    if(optimizer == "sgd"):
      self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
    elif(optimizer == "rmsprop"):
      self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr, alpha=self.alpha, eps=self.eps, weight_decay=self.weight_decay)
    elif(optimizer == "adam"):
      self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)



  def defineModel(self):
    self.model = nn.Sequential()

    inChannels = 3;     # RGB channels for inaturalist
    for i in range(len(self.num_filters)):
      self.model.append(nn.Conv2d(inChannels, self.num_filters[i], self.filter_sizes[i], padding=self.filter_sizes[i]//2))
      if self.isBatchNormalization:
        self.model.append(nn.BatchNorm2d(self.num_filters[i]))
      self.model.append(self.activationFun())
      self.model.append(nn.MaxPool2d(kernel_size=2))
      inChannels = self.num_filters[i]

    # computing flattened size
    input_shape = (3, 224, 224)
    with torch.no_grad():
      dummy_input = torch.zeros(1, *input_shape)
      dummy_output = self.model(dummy_input)
      flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

    self.model.append(nn.Flatten())
    self.model.append(nn.Linear(flattened_size, self.n_neurons_denseLayer))
    self.model.append(self.activationFun())

    if(self.dropout > 0):
      self.model.append(nn.Dropout(self.dropout))

    self.model.append(nn.Linear(self.n_neurons_denseLayer, self.num_classes))

  def forward(self, inputs):
    return self.model(inputs)

  def backward(self, outputs, labels):
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

  def updateWeights(self):
    self.optimizer.step()
