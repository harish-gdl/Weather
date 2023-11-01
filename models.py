import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    )

def get_model(model_name, num_features, num_classes, edge_index, edge_attr, x, mask=None):
    if model_name =="gcn":
        return GCN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    elif model_name == "gat":
        return GAT(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif model_name == "sage-gat":
        return SAGE_GAT(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Model {model_name}  is not available")


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=32, dropout=0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return torch.nn.functional.log_softmax(x, dim=1)

class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)

        return torch.nn.functional.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, dropout=0):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim,2)
        self.conv2 = GCNConv(2*hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

class SAGE_GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0):
        super(SAGE_GAT, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)
