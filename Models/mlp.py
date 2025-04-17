import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ========== 1. LOAD DATA ==========
train = pd.read_csv('/content/drive/MyDrive/dsa4263/2017_data.csv')
test = pd.read_csv('/content/drive/MyDrive/dsa4263/ddos2018_cleaned.csv').iloc[:, :-2]

# Keep only relevant columns
graph_cols = ['Source IP', 'Destination IP', 'Label']
feature_cols = [col for col in train.columns if col not in graph_cols + ['Flow ID', 'Timestamp']]

# Drop rows with NaNs for now
train = train.dropna(subset=graph_cols + feature_cols)
test = test.dropna(subset=graph_cols + feature_cols)

# Replace inf values
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(train[feature_cols]))
X_test = scaler.transform(imputer.transform(test[feature_cols]))

# Label encoding: BENIGN or Benign → 0, else 1
train['Label'] = train['Label'].map({'BENIGN': 0}).fillna(1).astype(int)
test['Label'] = test['Label'].map({'Benign': 0}).fillna(1).astype(int)

# ========== 2. BUILD GRAPH ==========
# Encode IPs as integer node indices
all_ips = pd.concat([train['Source IP'], train['Destination IP'],
                     test['Source IP'], test['Destination IP']]).unique()
ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}

# Create edge index tensor for train
edge_index_train = torch.tensor([
    [ip_to_idx[src] for src in train['Source IP']],
    [ip_to_idx[dst] for dst in train['Destination IP']]
], dtype=torch.long)

edge_attr_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(train['Label'].values, dtype=torch.long)

# PyG Data object for training
data_train = Data(edge_index=edge_index_train, edge_attr=edge_attr_train, y=y_train)

# ========== 3. DEFINE GNN ==========
class EdgeGNN(torch.nn.Module):
    def __init__(self, edge_feat_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(edge_feat_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, edge_attr):
        x = F.relu(self.fc1(edge_attr))
        return self.fc2(x)

# Instantiate model
model = EdgeGNN(edge_feat_dim=edge_attr_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# ========== 4. TRAIN ==========
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data_train.edge_attr)
    loss = loss_fn(out, data_train.y)
    loss.backward()
    optimizer.step()
    if epoch%50 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


from sklearn.metrics import classification_report, confusion_matrix

# ========== 5. PREPARE TEST GRAPH ==========
edge_index_test = torch.tensor([
    [ip_to_idx.get(src, -1) for src in test['Source IP']],
    [ip_to_idx.get(dst, -1) for dst in test['Destination IP']]
], dtype=torch.long)

# Filter out any edges with unknown IPs (-1)
valid_mask = (edge_index_test[0] != -1) & (edge_index_test[1] != -1)
edge_index_test = edge_index_test[:, valid_mask]
edge_attr_test = torch.tensor(X_test[valid_mask.numpy()], dtype=torch.float32)
y_test = torch.tensor(test['Label'].values[valid_mask.numpy()], dtype=torch.long)

# PyG Test Data
data_test = Data(edge_index=edge_index_test, edge_attr=edge_attr_test, y=y_test)

# ========== 6. EVALUATE ==========
model.eval()
with torch.no_grad():
    logits = model(data_test.edge_attr)
    preds = torch.argmax(logits, dim=1)

# Classification report
print("=== Evaluation Results ===")
print(classification_report(data_test.y.cpu(), preds.cpu(), digits=4))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(data_test.y.cpu(), preds.cpu()))

