import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import torch
from utils import (
    train_val_test_split,
    get_missing_feature_mask,
    get_mask,
    filling,
    FocalLoss,
)
import torch_sparse
from torch_scatter import scatter_add
from models import get_model
# from seeds import seeds
from data_loading import NepalWeatherGraphDATASET, get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
dataset_name = 'weather'
pred_type = 'temperature'
filling_method = "feature_propagation"
model_name = "gcn"
missing_rate = 0.0
num_iterations = 30
hidden_dim = 64
dropout = 0.2
lr = 0.001
epochs = 100
patience = 200
seed = 4258031


dataset = get_dataset(pred_type=pred_type)
n_nodes, n_features = dataset.data.x.shape
print("Number of nodes:",n_nodes)
print("Number of features:",n_features)
print("Number of classes:",dataset.num_classes)
print("Shape of edge_index:",dataset.data.edge_index.shape)
print("Shape of edge_attr:",dataset.data.edge_attr.shape)
print("Shape of label:",dataset.data.y.shape)
print("labels of nodes:",dataset.data.y)


num_classes = dataset.num_classes
data = train_val_test_split(
        seed=seed, data=dataset.data).to(device)

print("Length of train data:",data.x[data.train_mask].shape)
print("Length of val data:",data.x[data.val_mask].shape)
print("Length of test data:",data.x[data.test_mask].shape)

train_start = time.time()
missing_feature_mask = get_missing_feature_mask(
        rate=missing_rate, n_nodes=n_nodes, n_features=n_features).to(device)
x = data.x.clone()
x[~missing_feature_mask] = float("nan")
filled_features = filling(filling_method, data.edge_index, x, missing_feature_mask, num_iterations)
model = get_model(model_name=model_name,
                  num_features=data.num_features,
                  num_classes=num_classes,
                  edge_index=data.edge_index,
                  edge_attr=data.edge_attr,
                  x=x,
                  hidden_dim=hidden_dim,
                  dropout=dropout,
                  mask=missing_feature_mask,).to(device)

params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=lr)
# critereon = torch.nn.NLLLoss()
# critereon = torch.nn.CrossEntropyLoss()#FocalLoss
critereon = FocalLoss()
# Training loop
model.train()
x = torch.where(missing_feature_mask, data.x, filled_features)

train_loss_values = []  # List to store the training loss values
val_loss_values = []  # List to store the validation loss values
train_acc_values = []  # List to store the training accuracy values
val_acc_values = []  # List to store the validation accuracy values

for epoch in range(epochs):
    # x = torch.where(missing_feature_mask, data.x, filled_features)
    start = time.time()
    train_total_loss = 0
    val_total_loss = 0
    optimizer.zero_grad()
    train_out = model(x, data.edge_index,data.edge_attr)[data.train_mask]
    train_loss = critereon(train_out, data.y[data.train_mask])
    train_loss.backward()
    optimizer.step()
    train_total_loss += train_loss.item()

    _, train_pred = train_out.max(dim=1)
    train_correct = train_pred.eq(data.y[data.train_mask]).sum().item()
    train_total = data.train_mask.sum().item()
    train_acc = train_correct / train_total

    print(f"Epoch:{epoch + 1}/{epochs}")
    print(f"Train accuracy: {train_acc*100:.2f}%")
    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_out = model(x, data.edge_index,data.edge_attr)[data.val_mask]
        val_loss = critereon(val_out, data.y[data.val_mask])
        val_total_loss += val_loss.item()
    _, val_pred = val_out.max(dim=1)
    val_correct = val_pred.eq(data.y[data.val_mask]).sum().item()
    val_total = data.val_mask.sum().item()
    val_acc = val_correct / val_total

    print(f"Validation accuracy: {val_acc*100:.2f}%")
    print("===============================")

    train_loss_values.append(train_total_loss / len(data.train_mask))  # Append training loss value to the list
    val_loss_values.append(val_total_loss / len(data.val_mask))  # Append validation loss value to the list

    train_acc_values.append(train_acc)  # Append training accuracy value to the list
    val_acc_values.append(val_acc)  # Append validation accuracy value to the list
    if epoch > patience and max(val_acc_values[-patience:]) <= max(val_acc_values[: -patience]):
        print(f"Epoch {epoch + 1} - Train accuracy: {train_acc*100:.2f}%, Validation accuracy: {val_acc*100:.3f}%.")
        break

    print(f"The epoch {epoch + 1} took {time.time() - start:.2f} seconds.")

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_out = model(x, data.edge_index, data.edge_attr)[data.test_mask]
_, test_pred = test_out.max(dim=1)
test_correct = test_pred.eq(data.y[data.test_mask]).sum().item()
test_total = data.test_mask.sum().item()
test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Plot the training and validation loss curves
# plt.subplot(1, 2, 1)
plt.plot(train_loss_values, label='Training Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

# Plot the training and validation accuracy curves
# plt.subplot(1, 2, 2)
plt.plot(train_acc_values, label='Training Accuracy')
plt.plot(val_acc_values, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.legend()

# plt.tight_layout()
plt.show()
