import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader


def findOutputs(device, cnn, inputDataLoader, isTestData=False):
  cnn.eval()  # setting the model to evaluation model
  outputs = []
  total_loss = 0.0
  n_correct = 0
  n_correct_top5 = 0
  n_correct_top2 = 0
  n_samples = 0

  with torch.no_grad():
    for batch_idx, (x_batch, y_batch) in enumerate(inputDataLoader):
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      batch_outputs = cnn(x_batch)

      loss = nn.CrossEntropyLoss()(batch_outputs, y_batch)
      total_loss += loss.item() * x_batch.size(0)

      y_pred_batch = torch.argmax(batch_outputs, dim=1)
      n_correct += (y_pred_batch == y_batch).sum().item()
      n_samples += x_batch.size(0)

      if isTestData == True:
          y_pred_batch_top5 = torch.topk(batch_outputs, 5, dim=1).indices
          n_correct_top5 += y_pred_batch_top5.eq(y_batch.view(-1, 1)).sum().item()

          y_pred_batch_top2 = torch.topk(batch_outputs, 2, dim=1).indices
          n_correct_top2 += y_pred_batch_top2.eq(y_batch.view(-1, 1)).sum().item()
      outputs.append(batch_outputs)

  outputs = torch.cat(outputs)
  accuracy = (n_correct * 100.0) / n_samples
  avg_loss = total_loss / n_samples

  top5_accuracy = None
  top2_accuracy = None
  if isTestData == True:
      top5_accuracy = (n_correct_top5 * 100.0) / n_samples
      top2_accuracy = (n_correct_top2 * 100.0) / n_samples
  return outputs, accuracy, avg_loss, top5_accuracy, top2_accuracy
